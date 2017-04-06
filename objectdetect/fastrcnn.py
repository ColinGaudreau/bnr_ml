import theano
from theano import tensor as T

import numpy as np
import numpy.random as npr

import lasagne
from lasagne.layers import get_output, get_all_params
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import rmsprop, sgd

from skimage.io import imread
from skimage.transform import resize
from skimage import color
import cv2

from bnr_ml.utils.nonlinearities import smooth_l1
from bnr_ml.objectdetect.utils import BoundingBox
from bnr_ml.utils.helpers import meshgrid2D, format_image, StreamPrinter
from bnr_ml.objectdetect.detector import BaseDetector
from bnr_ml.objectdetect.nms import nms

from copy import deepcopy
from itertools import tee
import time
import pdb
from tqdm import tqdm

import dlib

from ml_logger.learning_objects import BaseLearningObject, BaseLearningSettings

class FastRCNNSettings(BaseLearningSettings):
	def __init__(
			self,
			train_annotations,
			test_annotations,
			train_args,
			print_obj=None,
			update_fn=rmsprop,
			update_args={'learning_rate': 1e-5},
			lmbda=1.,
			hyperparameters={}
		):
		super(FastRCNNSettings, self).__init__()
		self.train_annotations = train_annotations
		self.test_annotations = test_annotations
		self.train_args = train_args
		if print_obj is None:
			self.print_obj = StreamPrinter(open('/dev/stdout', 'w'))
		else:
			self.print_obj = print_obj
		self.update_fn = update_fn
		self.update_args = update_args
		self.lmbda = lmbda
		self.hyperparameters = hyperparameters

	def serialize(self):
		serialization = {}
		serialization['update_fn'] = self.update_fn.__str__()
		serialization['update_args'] = self.update_args
		serialization['lmbda'] = self.lmbda
		serialization['train_args'] = self.train_args
		serialization.update(self.hyperparameters)
		return serialization

class FastRCNNDetector(BaseLearningObject, BaseDetector):
	'''
		network should have an:
			input: this is the input layer
			detect: FC Layer for detections
			localize: FC Layer for localization

	'''
	def __init__(
		self,
		network,
		num_classes,
	):
		self.network = network
		self.num_classes = num_classes
		self.input = network['input'].input_var
		self.input_shape = network['input'].shape[2:]

		def reshape_loc_layer(loc_layer, num_classes):
			return loc_layer.reshape((-1, num_classes + 1, 4))

		self._detect = get_output(network['detect'], deterministic=False)
		self._detect_test = get_output(network['detect'], deterministic=True)
		self._localize = self._reshape_localization_layer(get_output(network['localize'], deterministic=False))
		self._localize_test = self._reshape_localization_layer(get_output(network['localize'], deterministic=True))

		# for detection
		self._trained = False
		self._hyperparameters = []

	def get_params(self):
		params, params_extra = get_all_params(self.network['detect']), get_all_params(self.network['localize'])
		for param in params_extra:
			if param not in params:
				params.append(param)
		return params

	def set_params(self, params):
		net_params = self.get_params()
		assert(params.__len__() == net_params.__len__())
		for p, v in zip(net_params, params):
			p.set_value(v)
		return
	
	def _reshape_localization_layer(self, localization_layer):
		return localization_layer.reshape((-1, self.num_classes + 1, 4))

	def _get_cost(self, detection_output, localization_output, target, lmbda=1., eps=1e-4):
		'''
		detection_output: NxK
		localization_output: NxKx4
		'''
		class_idx = target[:,-(self.num_classes + 1):].argmax(axis=1)
		mask = T.ones((target.shape[0], 1))
		mask = T.switch(T.eq(target[:,-(self.num_classes + 1):].argmax(axis=1), self.num_classes), 0, 1) # mask for non-object ground truth labels

		cost = categorical_crossentropy(detection_output, target[:,-(self.num_classes + 1):])
		if lmbda > 0:
			cost += lmbda * mask * T.sum(smooth_l1(localization_output[T.arange(localization_output.shape[0]), class_idx] - target[:,:4]), axis=1)
		
		return T.mean(cost)

	def get_weights(self):
		return [p.get_value() for p in self.get_params()]

	def get_hyperparameters(self):
		return self._hyperparameters

	def get_architecture(self):
		architecture = {}
		for layer in self.network:
			architecture[layer] = self.network[layer].__str__()
		return architecture

	def load_model(self, weights):
		self.set_params(weights)

	def train(self):
		# get settings for settings object
		train_annotations = self.settings.train_annotations
		test_annotations = self.settings.test_annotations
		train_args = self.settings.train_args
		print_obj = self.settings.print_obj
		update_fn = self.settings.update_fn
		update_args = self.settings.update_args
		lmbda = self.settings.lmbda

		self._trained = True
		target = T.matrix('target')
		
		# check if the training/testing functions have been compiled
		if not hasattr(self, '_train_fn') or not hasattr(self, '_test_fn'):
			print_obj.println('Getting cost...')
			cost = self._get_cost(self._detect, self._localize, target, lmbda=lmbda)
			cost_test = self._get_cost(self._detect_test, self._localize_test, target, lmbda=lmbda)
			
			if lmbda == 0:
				params = self.get_params()[:-2]
			else:
				params = self.get_params()
			updates = update_fn(cost, params, **update_args)

			ti = time.time();
			self._train_fn = theano.function([self.input, target], cost, updates=updates)
			print_obj.println('Compiling training function took %.3f seconds' % (time.time() - ti,))
			ti = time.time();
			self._test_fn = theano.function([self.input, target], cost_test)
			print_obj.println('Compiling test function took %.3f seconds' % (time.time() - ti,))

		print_obj.println('Beginning training')

		train_loss_batch = []
		test_loss_batch = []
		
		ti = time.time()
		for Xbatch, ybatch in generate_data(train_annotations, **train_args):
			err = self._train_fn(Xbatch, ybatch)
			train_loss_batch.append(err)
			print_obj.println('Batch error: %.4f' % err)
		
		for Xbatch, ybatch in generate_data(test_annotations, **train_args):
			test_loss_batch.append(self._test_fn(Xbatch, ybatch))

		train_loss = np.mean(train_loss_batch)
		test_loss = np.mean(test_loss_batch)

		print_obj.println('\n--------\nTrain Loss: %.4f, Test Loss: %.4f' % \
			(train_loss, test_loss))
		print_obj.println('Epoch took %.3f seconds.' % (time.time() - ti,))
		time.sleep(.01)
		
		return float(train_loss), float(test_loss)

	def _propose_regions(self, im, kvals, min_size):
		regions = []
		dlib.find_candidate_object_locations(im, regions, kvals, min_size)
		return [BoundingBox(r.left(), r.top(), r.right(), r.bottom()) for r in regions]

	def _filter_regions(self, regions, min_w, min_h):
		return [box for box in regions if box.w > min_w and box.h > min_h and box.isvalid()]

	def _rescale_image(self, im, max_dim):
		scale = float(max_dim) / (max(im.shape[0], im.shape[1]))
		new_shape = (int(im.shape[1] * scale), int(im.shape[0] * scale))
		return cv2.resize(im, new_shape, interpolation=cv2.INTER_LINEAR)

	def detect(
			self,
			im,
			thresh=0.5,
			kvals=(30,200,3),
			max_dim=600,
			min_w=10,
			min_h=10,
			min_size=100,
			batch_size=50,
			max_regions=1000,
			overlap=.4,
			num_to_label=None,
		):
		if im.shape.__len__() == 2:
			im = np.repeat(im.reshape(im.shape + (1,)), 3, axis=2)
		if im.shape[2] > 3:
			im = im[:,:,:3]
		if im.max() > 1:
			im = im / 255.
		if im.dtype != theano.config.floatX:
			im = im.astype(theano.config.floatX)
		
		old_size = im.shape[:2]
		im = self._rescale_image(im, max_dim)

		# compile detection function if it has not yet been done
		detect_input_ndarray = np.zeros((batch_size,3) + self.input_shape, dtype=theano.config.floatX)
		if self._trained or not hasattr(self, '_detect_fn'):
			self._detect_input = theano.shared(detect_input_ndarray, name='detection_input', borrow=True)
			self.network['input'].input_var = self._detect_input
			detection = get_output(self.network['detect'], deterministic=True)
			localization = self._reshape_localization_layer(get_output(self.network['localize'], deterministic=True))
			self.network['input'].input_var = self.input

			# need to add stuff to connect all of this
			self._detect_fn = theano.function([], [detection, localization])
			self._trained = False

		regions = np.asarray(self._filter_regions(self._propose_regions(im, kvals, min_size), min_w, min_h))
		if max_regions is not None:
			max_regions = min(regions.__len__(), max_regions)
			regions = regions[npr.choice(regions.__len__(), max_regions, replace=False)]
		
		swap = lambda im: im.swapaxes(2,1).swapaxes(1,0)
		# im_list = np.zeros((regions.__len__(), 3) + self.input_shape, dtype=theano.config.floatX)

		subim_ph = np.zeros(self.input_shape + (3,), dtype=theano.config.floatX)
		batch_index = 0
		class_score = np.zeros((regions.__len__(), self.num_classes + 1), dtype=theano.config.floatX)
		coord = np.zeros((regions.__len__(), self.num_classes + 1, 4), dtype=theano.config.floatX)
		for i, box in enumerate(regions):
			subim = box.subimage(im)
			cv2.resize(subim, self.input_shape[::-1], dst=subim_ph, interpolation=cv2.INTER_NEAREST)

			detect_input_ndarray[batch_index] = swap(subim_ph)
			batch_index += 1

			if batch_index == batch_size:
				self._detect_input.set_value(detect_input_ndarray, borrow=True)
				class_score[i - (batch_size - 1):i + 1], coord[i - (batch_size - 1):i + 1] = self._detect_fn()
				batch_index = 0

		if batch_index != batch_size and batch_index != 0:
			self._detect_input.set_value(detect_input_ndarray[:batch_index], borrow=True)
			class_score[i - batch_index:i], coord[i - batch_index:i] = self._detect_fn()

		# compute scores for the regions
		# if batch_size is not None:
		# 	class_score, coord = np.zeros((regions.__len__(), self.num_classes + 1)), np.zeros((regions.__len__(), self.num_classes + 1, 4))
		# 	for i in range(0, regions.__len__(), batch_size):
		# 		class_score[i:i+batch_size], coord[i:i+batch_size] = self._detect_fn(im_list[i:i+batch_size])
		# else:
		# 	class_score, coord = self._detect_fn(im_list)

		# filter out windows which are 1) not labeled to be an object 2) below threshold
		class_id = np.argmax(class_score[:,:-1], axis=1)
		class_score = class_score[np.arange(class_score.shape[0]), class_id]
		is_obj = class_score > thresh
		coord = coord[np.arange(coord.shape[0]), class_id][is_obj]
		coord[:,2:] = np.exp(coord[:,2:])
		coord[:,:2] -= coord[:,2:]/2
		regions = [box for (o,box) in zip(is_obj, regions) if o]

		# re-adjust boxes for the image
		objects = []
		scale_factor = (float(old_size[0])/im.shape[0], float(old_size[1])/im.shape[1])
		for i, box in enumerate(regions):
			coord[i, [0,2]] *= box.w
			coord[i, [1,3]] *= box.h
			coord[i, 0] += box.xi
			coord[i, 1] += box.yi
			coord[i, 2:] += coord[i, :2]
			obj = BoundingBox(*coord[i,:].tolist())
			obj *= scale_factor
			objects.append(obj)

		objects = np.asarray(objects)
		class_score, class_id = class_score[is_obj], class_id[is_obj]

		# convert to expected format for detector output
		output = {}
		for cls in np.unique(class_id):
			cls_output = {}
			cls_idx = class_id == cls
			boxes, scores = nms(objects[cls_idx].tolist(), scores=class_score[cls_idx].tolist(), overlap=overlap)
			cls_output['boxes'] = boxes
			cls_output['scores'] = scores
			if num_to_label is not None:
				cls = num_to_label[cls]
			output[cls] = cls_output

		return output

def format_boxes(annotation):
	boxes = np.zeros((annotation.__len__(),4))
	for i in range(boxes.shape[0]):
		boxes[i] = [annotation[i]['x'], annotation[i]['y'], annotation[i]['w'], annotation[i]['h']]
	return boxes

def generate_proposal_boxes(boxes, n_proposals=1000):
	'''
	Generate proposal regions using boxes; boxes should be 
	an Nx4 matrix, were boxes[i] = [x,y,w,h]
	
	N - number of proposals per box
	'''
	
	proposals = np.zeros((boxes.shape[0] * n_proposals, 4))
	proposal = np.zeros((n_proposals, 4))
	n_pos = int(0.25 * n_proposals)
	n_neg = n_proposals - n_pos
	for i in range(boxes.shape[0]):
		# positive box examples
		proposal[:n_pos,0] = (boxes[i,0] - boxes[i,2]/5) + (2./5) * boxes[i,2] * npr.rand(n_pos)
		proposal[:n_pos,1] = (boxes[i,1] - boxes[i,3]/5) + (2./5) * boxes[i,3] * npr.rand(n_pos)
		proposal[:n_pos,2] = (4./5) * boxes[i,2] + (2./5) * boxes[i,2] * npr.rand(n_pos)
		proposal[:n_pos,3] = (4./5) * boxes[i,3] + (2./5) * boxes[i,3] * npr.rand(n_pos)
		# negative examples
		proposal[n_pos:,0] = (boxes[i,0] - boxes[i,2]/2) + boxes[i,2] * npr.rand(n_neg)
		proposal[n_pos:,1] = (boxes[i,1] - boxes[i,3]/2) + boxes[i,3] * npr.rand(n_neg)
		proposal[n_pos:,2] = boxes[i,2]/2 + (3./2 * boxes[i,2]) * npr.rand(n_neg)
		proposal[n_pos:,3] = boxes[i,3]/2 + (3./2 * boxes[i,3]) * npr.rand(n_neg)
		
		proposals[i*n_proposals:(i+1)*n_proposals] = proposal
	
	return proposals

def find_valid_boxes(boxes, proposals):
	'''
	from the proposals, find the valid negative/positive examples
	'''
	boxes = boxes.reshape((1,) + boxes.shape)
	proposals = proposals.reshape(proposals.shape + (1,))
	
	# calculate iou
	xi = np.maximum(boxes[:,:,0], proposals[:,0])
	yi = np.maximum(boxes[:,:,1], proposals[:,1])
	xf = np.minimum(boxes[:,:,[0,2]].sum(axis=2), proposals[:,[0,2]].sum(axis=1))
	yf = np.minimum(boxes[:,:,[1,3]].sum(axis=2), proposals[:,[1,3]].sum(axis=1))
	
	w, h = np.maximum(xf - xi, 0.), np.maximum(yf - yi, 0.)
	
	isec = w * h
	union = boxes[:,:,2:].prod(axis=2) + proposals[:,2:].prod(axis=1) - isec
	
	iou = isec / union
	
	overlap = proposals[:,2:].prod(axis=1) / union
	
	neg_idx = np.bitwise_and(
		np.bitwise_and(
			iou > 0.1, 
			iou < 0.5
		),
		overlap < 0.3
	) 
	
	pos_idx = iou > 0.5
	
	# get any box which doesn't overlap with any box
	neg_idx = np.sum(neg_idx, axis=1) >= 1
	
	# filter out boxes that have an iou > .5 with more than one object
	pos_idx = np.sum(pos_idx, axis=1) == 1
	
	indices = np.arange(proposals.shape[0])
	
	# get object index for matched object
	obj_idx = iou[pos_idx,:].argmax(axis=1)
	
	return indices[neg_idx], indices[pos_idx], obj_idx

def colour_space_augmentation(im):
	im = color.rgb2hsv(im)
	im[:,:,2] *= (0.2 * npr.rand() + 0.9)
	idx = im[:,:,2] > 1.0
	im[:,:,2][idx] = 1.0
	im = color.hsv2rgb(im)
	return im

def generate_example(
		im,
		input_shape,
		num_classes,
		label_to_num,
		annotation,
		proposals,
		indices,
		n_neg,
		n_pos
	):
	neg_idx, pos_idx, obj_idx = indices

	if neg_idx.size == 0 or pos_idx.size == 0:
		print('Warning, no valid prosals were given.')
		return None
	
	neg_examples = proposals[neg_idx,:]
	
	X = np.zeros((n_neg + n_pos, 3) + input_shape, dtype=theano.config.floatX)
	y = np.zeros((n_neg + n_pos, 4 + num_classes + 1), dtype=theano.config.floatX)
	
	try:
		pos_choice = npr.choice(np.arange(pos_idx.size), size=n_pos, replace=False)
	except:
		print('Warning, positive examples sampled with replacement from proposal boxes.')
		pos_choice = npr.choice(np.arange(pos_idx.size), size=n_pos, replace=True)
	try:
		neg_choice = npr.choice(np.arange(neg_idx.size), size=n_neg, replace=False)
	except:
		print('Warning, positive examples sampled with replacement from proposal boxes.')
		neg_choice = npr.choice(np.arange(neg_idx.size), size=n_neg, replace=True)
	
	neg_idx, pos_idx, obj_idx = neg_idx[neg_choice], pos_idx[pos_choice], obj_idx[pos_choice]
	
	# generate negative examples
	cls = np.zeros(num_classes + 1)
	cls[-1] = 1.
	coord = np.asarray([0.,0.,1.,1.])
	for i in range(n_neg):
		xi, yi = int(max(0,proposals[neg_idx[i],0])), int(max(0,proposals[neg_idx[i],1]))
		xf = int(min(im.shape[1], proposals[neg_idx[i],[0,2]].sum()))
		yf = int(min(im.shape[0], proposals[neg_idx[i],[1,3]].sum()))
		subim = colour_space_augmentation(resize(im[yi:yf,xi:xf], input_shape))
		y[i,:4] = coord
		y[i,-(num_classes + 1):] = cls
		X[i] = subim.swapaxes(2,1).swapaxes(1,0)
	
	cls[-1] = 0.
	# generate positive examples
	for i in range(n_pos):
		xi, yi = int(max(0,proposals[pos_idx[i],0])), int(max(0,proposals[pos_idx[i],1]))
		xf = int(min(im.shape[1], proposals[pos_idx[i],[0,2]].sum()))
		yf = int(min(im.shape[0], proposals[pos_idx[i],[1,3]].sum()))
		subim = colour_space_augmentation(resize(im[yi:yf,xi:xf], input_shape))
		coord[0] = (annotation[obj_idx[i]]['x'] + annotation[obj_idx[i]]['w']/2 - proposals[pos_idx[i],0]) / proposals[pos_idx[i],2] # get center of object
		coord[1] = (annotation[obj_idx[i]]['y'] + annotation[obj_idx[i]]['h']/2 - proposals[pos_idx[i],1]) / proposals[pos_idx[i],3]
		coord[2] = np.log(float(annotation[obj_idx[i]]['w']) / proposals[pos_idx[i], 2])
		coord[3] = np.log(float(annotation[obj_idx[i]]['h']) / proposals[pos_idx[i], 3])
		cls[label_to_num[annotation[obj_idx[i]]['label']]] = 1.
		X[i+n_neg] = subim.swapaxes(2,1).swapaxes(1,0)
		y[i+n_neg,:4], y[i+n_neg,-(num_classes+1):] = coord, cls
	
	return X, y

def generate_data(
		annotations,
		input_shape=None,
		num_classes=None,
		label_to_num=None,
		n_neg=9,
		n_pos=3,
		batch_size=2,
		n_proposals=2000
	):
	if not isinstance(annotations, np.ndarray):
		annotations = np.asarray(annotations)
	npr.shuffle(annotations)
	n_total = n_neg + n_pos
	cnt = 0

	for i in range(annotations.size):
		X = np.zeros((n_total * batch_size, 3) + input_shape, dtype=theano.config.floatX)
		y = np.zeros((n_total * batch_size, 4 + num_classes + 1), dtype=theano.config.floatX)
		boxes = format_boxes(annotations[i]['annotations'])
		proposals = generate_proposal_boxes(boxes, n_proposals=n_proposals)
		indices = find_valid_boxes(boxes, proposals)
		im = format_image(imread(annotations[idx]['image']), dtype=theano.config.floatX)
		data = generate_example(im, input_shape, num_classes, label_to_num, annotations[idx]['annotations'], proposals, indices, n_neg, n_pos)
		if data is not None:
			X[cnt*n_total:(cnt+1)*n_total], y[cnt*n_total:(cnt+1)*n_total] = data[0], data[1]
			cnt += 1

		if cnt == batch_size or (i-1) == annotations.size:
			X, y = X[:cnt*n_total], y[:cnt*n_total]
			if X.shape[0] > 0:
				idx = np.arange(X.shape[0])
				npr.shuffle(idx)
				yield X[idx], y[idx]
			cnt = 0
