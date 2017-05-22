import theano
from theano import tensor as T
import numpy as np
import numpy.random as npr

from bnr_ml.utils.helpers import StreamPrinter, meshgrid, format_image
from bnr_ml.utils.nonlinearities import softmax, smooth_abs, smooth_l1
from bnr_ml.objectdetect.utils import BoundingBox
from bnr_ml.objectdetect.nms import nms
from bnr_ml.logger.learning_objects import BaseLearningObject, BaseLearningSettings

from lasagne import layers
from lasagne.updates import rmsprop

import cv2

import time
import pdb

class SSDSettings(BaseLearningSettings):
	def __init__(
			self,
			gen_fn,
			train_annotations,
			test_annotations,
			train_args,
			test_args=None,
			print_obj=None,
			update_fn=rmsprop,
			update_args={'learning_rate': 1e-5},
			alpha=1.0,
			min_iou=0.5,
			hyperparameters={}
		):
		super(SSDSettings, self).__init__()
		self.gen_fn = gen_fn
		self.train_annotations = train_annotations
		self.test_annotations = test_annotations
		self.train_args = train_args
		if test_args is None:
			self.test_args = train_args
		else:
			self.test_args = test_args
		if print_obj is None:
			self.print_obj = StreamPrinter(open('/dev/stdout', 'w'))
		else:
			self.print_obj = print_obj
		self.update_fn = update_fn
		self.update_args = update_args
		self.alpha = alpha
		self.min_iou = min_iou
		self.hyperparameters = hyperparameters

	def serialize(self):
		serialization = {}
		serialization['update_fn'] = self.update_fn.__str__()
		serialization['update_args'] = self.update_args
		serialization['alpha'] = self.alpha
		serialization['min_iou'] = self.min_iou
		serialization.update(self.hyperparameters)
		return serialization

class SingleShotDetector(BaseLearningObject):
	def __init__(
		self,
		network,
		num_classes,
		ratios=[(1,1),(1./np.sqrt(2),np.sqrt(2)),(np.sqrt(2),1./np.sqrt(2)),(1./np.sqrt(3),np.sqrt(3)),(np.sqrt(3),1./np.sqrt(3)),(1.2,1.2)],
		smin=0.2,
		smax=0.95
	):
		assert('detection' in network and 'input' in network)
		super(SingleShotDetector, self).__init__()	
		self.network = network
		self.num_classes = num_classes
		self.ratios = ratios
		self.smin = smin
		self.smax = smax
		self.input = network['input'].input_var
		self.input_shape = network['input'].shape[-2:]
		self._hyperparameters = [{'ratios': ratios, 'smin': smin, 'smax': smax}]

		# build default map
		self._build_default_maps()
		self._build_predictive_maps()

		self._trained = True
		self._recompile = True

		return
	
	def get_params(self):
		parameters = []
		for lname in self.network:
			if lname != 'detection':
				parameters.extend(self.network[lname].get_params())
			else:
				for dlayer in self.network['detection']:
					parameters.extend(dlayer.get_params())
		return parameters

	def set_params(self, params):
		net_params = self.get_params()
		assert(params.__len__() == net_params.__len__())
		for p, v in zip(net_params, params):
			p.set_value(v)
		return

	'''
	Implement funciton for the BaseLearningObject class
	'''
	def get_weights(self):
		return [p.get_value() for p in self.get_params()]

	def get_hyperparameters(self):
		self._hyperparameters = super(SingleShotDetector, self).get_hyperparameters()
		return self._hyperparameters

	def get_architecture(self):
		architecture = {}
		return architecture

	def load_model(self, weights):
		self.set_params(weights)
		# self._build_predictive_maps()

	def train(self):
		self._trained = True

		# get settings from SSD settings object
		gen_fn = self.settings.gen_fn
		train_annotations = self.settings.train_annotations
		test_annotations = self.settings.test_annotations
		train_args = self.settings.train_args
		test_args = self.settings.test_args
		print_obj = self.settings.print_obj
		update_fn = self.settings.update_fn
		update_args = self.settings.update_args
		alpha = self.settings.alpha
		min_iou = self.settings.min_iou

		if not hasattr(self, '_train_fn') or not hasattr(self, '_test_fn'):
			if not hasattr(self, 'target'):
				self.target = T.tensor3('target')

			print_obj.println('Getting cost...')

			# get cost
			ti = time.time()
			cost = self._get_cost(self.input, self.target, alpha=alpha, min_iou=min_iou)

			print_obj.println('Creating cost variable took %.4f seconds' % (time.time() - ti,))

			parameters = self.get_params()
			grads = T.grad(cost, parameters)
			updates = update_fn(grads, parameters, **update_args)

			print_obj.println('Compiling...')

			ti = time.time()
			self._train_fn = theano.function([self.input, self.target], cost, updates=updates)
			self._test_fn = theano.function([self.input, self.target], cost)

			print_obj.println('Compiling functions took %.4f seconds' % (time.time() - ti,))

		print_obj.println('Beginning training...')

		train_loss_batch, test_loss_batch = [], []

		for Xbatch, ybatch in gen_fn(train_annotations, **train_args):
			err = self._train_fn(Xbatch, ybatch)
			train_loss_batch.append(err)
			print_obj.println('Batch error: %.4f' % (err,))

		for Xbatch, ybatch in gen_fn(test_annotations, **test_args):
			test_loss_batch.append(self._test_fn(Xbatch, ybatch))

		train_loss = np.mean(train_loss_batch)
		test_loss = np.mean(test_loss_batch)

		print_obj.println('\n------\nTrain Loss: %.4f, Test Loss: %.4f\n' % (train_loss, test_loss))

		return train_loss, test_loss

	def detect(self, im, thresh=0.75, overlap=0.4, n_apply=1, num_to_label=None):
		old_size = im.shape[:2]
		im = cv2.resize(im, self.input_shape[::-1], interpolation=cv2.INTER_NEAREST)
		im = format_image(im, theano.config.floatX)
		swap = lambda im: im.swapaxes(2,1).swapaxes(1,0).reshape((1,3) + self.input_shape)

		if not (self._trained and hasattr(self, '_detect_fn')):
			self._detect_fn = theano.function([self.input], self._predictive_maps)

		detections = self._detect_fn(swap(im))

		boxes = []
		for detection, dmap in zip(detections, self._default_maps_asarray):
			for i in range(detection.shape[1]):
				for j in range(detection.shape[3]):
					for k in range(detection.shape[4]):
						coord, score = detection[0,i,:4,j,k], detection[0,i,-(self.num_classes + 1):-1,j,k]
						coord[2:] = dmap[i,2:4,j,k] * np.exp(coord[2:])
						coord[:2] = dmap[i,2:4,j,k] * coord[:2] + dmap[i,:2,j,k]
						if score.max() > thresh:
							cls = score.argmax()
							if num_to_label is not None:
								cls = num_to_label[cls]
							box = BoundingBox(coord[0], coord[1], coord[0] + coord[2], coord[1] + coord[3]) * old_size
							box.cls = cls
							box.confidence = score.max()
							boxes.append(box)
	
		boxes = nms(boxes, overlap=overlap, n_apply=n_apply)'scores': scores

		return boxes
	
	def _build_default_maps(self):
		'''
		Get matrix with default boxes for each of 
		the feature maps
		'''
		default_maps = []
		default_maps_asarray = []
		fms = self.network['detection']
		
		for i, fm in enumerate(fms):
			shape = layers.get_output_shape(fm)[-2:]
			fmap = np.zeros((self.ratios.__len__(), 4) + shape, dtype=theano.config.floatX)
			
			xcoord, ycoord = np.linspace(0, 1, shape[1] + 1), np.linspace(0, 1, shape[0] + 1)
			if shape[1] > 1 and shape[1] > 1:
				# xcoord, ycoord = xcoord[:-1], ycoord[:-1]
				xcoord, ycoord = (xcoord[:-1] + xcoord[1:])/2, (ycoord[:-1] + ycoord[1:])/2
			else:
				# xcoord, ycoord = np.asarray([0.]), np.asarray([0.])
				xcoord, ycoord = np.asarray([0.5]), np.asarray([0.5])
	
			xcoord, ycoord = np.meshgrid(xcoord, ycoord)
			
			# set coordinates
			fmap[:,0,:,:] = xcoord.reshape((1,) + shape)
			fmap[:,1,:,:] = ycoord.reshape((1,) + shape)
			
			# set scale
			scale = self.smin + (self.smax - self.smin)/(fms.__len__() - 1) * i
			for j, ratio in enumerate(self.ratios):
				fmap[j,2,:,:] = float(ratio[0]) * scale
				fmap[j,3,:,:] = float(ratio[1]) * scale
			
			# shift xi, yi
			fmap[:,0,:,:] -= (fmap[:,2,:,:]/2)
			fmap[:,1,:,:] -= (fmap[:,3,:,:]/2)

			default_maps_asarray.append(fmap)
			fmap = theano.shared(fmap, name='map_{}'.format(i), borrow=True)
			default_maps.append(fmap)
		
		self._default_maps = default_maps
		self._default_maps_asarray = default_maps_asarray

	def _build_predictive_maps(self):
		'''
		Reshape detection layers and set nonlinearities.
		'''
		predictive_maps = []
		fms = self.network['detection']
		
		for i, fm in enumerate(fms):
			dmap = self._default_maps[i]
			fmap = layers.get_output(fm)
			shape = layers.get_output_shape(fm)[2:]
			fmap = T.reshape(fmap, (-1, self.ratios.__len__(), 4 + (self.num_classes + 1)) + shape)
			fmap = T.set_subtensor(fmap[:,:,-(self.num_classes + 1):,:,:], softmax(fmap[:,:,-(self.num_classes + 1):,:,:], axis=2))
			#fmap = T.set_subtensor(fmap[:,:,:2,:,:], fmap[:,:,:2,:,:] + dmap[:,:2].dimshuffle('x',0,1,2,3)) # offset due to default box
			predictive_maps.append(fmap)
		
		self._predictive_maps = predictive_maps
	
	def _get_iou(self, mat1, mat2):
		'''
		mat1/mat2 should be an N x M x L x P x Q x S[0] x S[1],
		N - size of batch
		M - max number of objects in image
		L - number of default boxes
		P - 4 + num_classes
		S - feature map shape
		
		returns mat minus the 4th dimension
		'''
		xi = T.maximum(mat1[:,:,:,0], mat2[:,:,:,0])
		yi = T.maximum(mat1[:,:,:,1], mat2[:,:,:,1])
		xf = T.minimum(mat1[:,:,:,[0,2]].sum(axis=3), mat2[:,:,:,[0,2]].sum(axis=3))
		yf = T.minimum(mat1[:,:,:,[1,3]].sum(axis=3), mat2[:,:,:,[1,3]].sum(axis=3))
		
		w, h = T.maximum(xf - xi, 0), T.maximum(yf - yi, 0)
		
		isec = w * h
		union = mat1[:,:,:,2:4].prod(axis=3) + mat2[:,:,:,2:4].prod(axis=3) - isec
		
		iou = T.maximum(isec / union, 0.)
		return iou
	
	def _get_cost(self, input, truth, alpha=1., min_iou=0.5):
		cost = 0.
		
		# create ground truth for non-object class
		neg_example = theano.shared(np.zeros(self.num_classes + 1, dtype=theano.config.floatX))
		neg_example = T.set_subtensor(neg_example[-1], 1.)
		neg_example = neg_example.dimshuffle('x','x',0,'x','x')
		
		for i in range(self._predictive_maps.__len__()):
			dmap = self._default_maps[i]
			fmap = self._predictive_maps[i]
			shape = layers.get_output_shape(self.network['detection'][i])[2:]
			
			# get iou between default maps and ground truth
			iou_default = self._get_iou(dmap.dimshuffle('x','x',0,1,2,3), truth.dimshuffle(0,1,'x',2,'x','x'))
			#pdb.set_trace()
			# get which object for which cell
			idx_match = T.argmax(iou_default, axis=1)

			# extend truth to cover all cell/box/examples
			truth_extended = T.repeat(
				T.repeat(
					T.repeat(truth.dimshuffle(0,1,'x',2,'x','x'), self.ratios.__len__(), axis=2), 
					shape[0], axis=4
				), 
				shape[1], axis=5
			)

			idx1, idx2, idx3, idx4 = meshgrid(
				T.arange(truth.shape[0]),
				T.arange(self.ratios.__len__()),
				T.arange(shape[0]),
				T.arange(shape[1])
			)

			# copy truth for every cell/box.
			truth_extended = truth_extended[idx1, idx_match, idx2, :, idx3, idx4].dimshuffle(0,1,4,2,3)
			
			iou_default = iou_default.max(axis=1)

			iou_gt_min = iou_default >= min_iou

			dmap_extended = dmap.dimshuffle('x',0,1,2,3)
			
			# penalize coordinates
			cost_fmap = 0.

			cost_fmap += (smooth_l1((fmap[:,:,0] - (truth_extended[:,:,0] - dmap_extended[:,:,0]) / dmap_extended[:,:,2])[iou_gt_min.nonzero()])).sum()
			cost_fmap += (smooth_l1((fmap[:,:,1] - (truth_extended[:,:,1] - dmap_extended[:,:,1]) / dmap_extended[:,:,3])[iou_gt_min.nonzero()])).sum()
			cost_fmap += (smooth_l1((fmap[:,:,2] - T.log(truth_extended[:,:,2] / dmap_extended[:,:,2]))[iou_gt_min.nonzero()])).sum()
			cost_fmap += (smooth_l1((fmap[:,:,3] - T.log(truth_extended[:,:,3] / dmap_extended[:,:,3]))[iou_gt_min.nonzero()])).sum()

			class_cost = -(truth_extended[:,:,-(self.num_classes + 1):] * T.log(fmap[:,:,-(self.num_classes + 1):])).sum(axis=2)
			cost_fmap += (alpha * class_cost[iou_gt_min.nonzero()].sum())

			# find negative examples
			iou_default = iou_default.reshape((-1,))
			iou_idx_sorted = T.argsort(iou_default)[::-1]
			iou_st_min = iou_default < min_iou
			
			# Choose index for top boxes whose overlap is smaller than the min overlap.
			pos_size = iou_gt_min[iou_gt_min.nonzero()].size
			neg_size = pos_size * 3 # ratio of 3 to 1
			#neg_size = 10
			iou_idx_sorted = iou_idx_sorted[iou_st_min[iou_idx_sorted].nonzero()][:neg_size]
			neg_size = iou_idx_sorted.size

			neg_size, pos_size = T.maximum(1., neg_size), T.maximum(1., pos_size)

			# Add the negative examples to the costs.
			neg_class_cost = -(neg_example * T.log(fmap[:,:,-(self.num_classes + 1):])).sum(axis=2).reshape((-1,))

			cost_fmap /= pos_size
			cost_fmap += (alpha * neg_class_cost[iou_idx_sorted].sum() / neg_size)
			
			# normalize
			# cost_fmap /= T.maximum(1., pos_size + neg_size)
			
			cost += cost_fmap

		return cost
