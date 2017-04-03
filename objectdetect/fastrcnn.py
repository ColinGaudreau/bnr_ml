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
from bnr_ml.utils.helpers import meshgrid2D, format_image
from bnr_ml.objectdetect.detector import BaseDetector
from bnr_ml.objectdetect.nms import nms

from copy import deepcopy
from itertools import tee
import time
import pdb
from tqdm import tqdm

import dlib

from ml_logger.learning_objects import BaseLearningObject

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

	def train(
			self,
			train_annotations=None,
			test_annotations=None,
			label_dict=None,
			print_obj=None,
			update_fn=rmsprop,
			num_batch=2,
			N=20,
			neg=.5,
			num_batch_test=5,
			N_test=10,
			neg_test=.5,
			lr=1e-4,
			momentum=0.9,
			lmbda=1.,
			hyperparameters={},
		):
		'''
		'''
		# update hyperparameters list
		hyperparameters.update({
			'num_batch': num_batch,
			'N': N,
			'neg': neg,
			'lr': lr,
			'momentum': momentum,
			'lambda': lmbda,
			'update_fn': update_fn.__str__()
		})
		self._hyperparameters.append(hyperparameters)

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
			updates = rmsprop(cost, params, learning_rate=lr)

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
		for Xbatch, ybatch in generate_rois(train_annotations, self.input_shape, self.num_classes, label_dict, num_batch=num_batch, N=N, neg=neg):
			err = self._train_fn(Xbatch, ybatch)
			train_loss_batch.append(err)
			print_obj.println('Batch error: %.4f' % err)
		
		for Xbatch, ybatch in generate_rois(test_annotations, self.input_shape, self.num_classes, label_dict, num_batch=num_batch_test, N=N_test, neg=neg_test):
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
		proposal[:n_pos,0] = (boxes[i,0] - boxes[i,2]/4) + (2./4) * boxes[i,2] * npr.rand(n_pos)
		proposal[:n_pos,1] = (boxes[i,1] - boxes[i,3]/4) + (2./4) * boxes[i,3] * npr.rand(n_pos)
		proposal[:n_pos,2] = (3./4) * boxes[i,2] + (2./4) * boxes[i,2] * npr.rand(n_pos)
		proposal[:n_pos,3] = (3./4) * boxes[i,3] + (2./4) * boxes[i,3] * npr.rand(n_pos)
		
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
		coord[0] = (annotation[obj_idx[i]]['x'] - proposals[pos_idx[i],0]) / proposals[pos_idx[i],2]
		coord[1] = (annotation[obj_idx[i]]['y'] - proposals[pos_idx[i],1]) / proposals[pos_idx[i],3]
		coord[2] = np.log(float(annotation[obj_idx[i]]['w']) / proposals[pos_idx[i], 2])
		coord[3] = np.log(float(annotation[obj_idx[i]]['h']) / proposals[pos_idx[i], 3])
		cls[label_to_num[annotation[obj_idx[i]]['label']]] = 1.
		X[i+n_neg] = subim.swapaxes(2,1).swapaxes(1,0)
		y[i+n_neg,:4], y[i+n_neg,-(num_classes+1):] = coord, cls
	
	return X, y

def generate_data(
		annotations,
		input_shape,
		num_classes,
		label_to_num,
		n_neg,
		n_pos,
		batch_size,
		n_proposals=1500
	):
	if not isinstance(annotations, np.ndarray):
		annotations = np.asarray(annotations)
	npr.shuffle(annotations)
	
	for i in range(0,annotations.size,batch_size):
		X = np.zeros(((n_neg + n_pos) * batch_size, 3) + input_shape, dtype=theano.config.floatX)
		y = np.zeros(((n_neg + n_pos) * batch_size, 4 + num_classes + 1), dtype=theano.config.floatX)
		for j in range(min(batch_size, annotations.size - i)):
			idx = i + j
			boxes = format_boxes(annotations[idx]['annotations'])
			proposals = generate_proposal_boxes(boxes, N=n_proposals)
			indices = find_valid_boxes(boxes, proposals)
			im = format_image(imread(annotations[idx]['image']), dtype=theano.config.floatX)
			X_j, y_j = generate_example(im, input_shape, num_classes, label_to_num, annotations[idx]['annotations'], proposals, indices, n_neg, n_pos)
			X[j*(n_neg + n_pos):(j+1)*(n_neg + n_pos)] = X_j
			y[j*(n_neg + n_pos):(j+1)*(n_neg + n_pos)] = y_j
		X, y = X[:(j+1)*(n_neg + n_pos)], y[:(j+1)*(n_neg + n_pos)]
		idx = np.arange(X.shape[0])
		npr.shuffle(idx)
		yield X[idx], y[idx]

# def _gen_boxes(imsize, num_pos=20, num_scale=20):
# 	x = np.linspace(0, imsize[1], num_pos)
# 	y = np.linspace(0, imsize[0], num_pos)
# 	w = np.linspace(0, imsize[1], num_scale)
# 	h = np.linspace(0, imsize[0], num_scale)
	
# 	x,y,w,h = np.meshgrid(x,y,w,h)
# 	x,y,w,h = x.reshape((-1,1)), y.reshape((-1,1)), w.reshape((-1,1)), h.reshape((-1,1))
# 	boxes = np.concatenate((x,y,w,h), axis=1)

# 	boxes[:,0] = np.maximum(0., np.minimum(imsize[0], boxes[:,0]))
# 	boxes[:,1] = np.maximum(0., np.minimum(1, boxes[:,1]))
# 	idx = boxes[:,0] + boxes[:,2] > imsize[1]
# 	boxes[idx,2] = imsize[1]
# 	idx = boxes[:,1] + boxes[:,3] > imsize[0]
# 	boxes[idx,3] = imsize[0]

# 	return boxes

# def _generate_boxes_from_obj(obj, imsize, num_pos=20, num_scale=20, mult=2):
# 	x = np.linspace(max(0,obj['x'] - .7*obj['w']), min(imsize[1], obj['x'] + .7*obj['w']), num_pos)
# 	y = np.linspace(max(0,obj['y'] - .7*obj['h']), min(imsize[0], obj['y'] + .7*obj['h']), num_pos)
# 	w = np.linspace(obj['w'] / mult, min(imsize[1], obj['w'] * mult), num_pos)
# 	h = np.linspace(obj['h'] / mult, min(imsize[0], obj['h'] * mult), num_pos)
		
# 	x,y,w,h = np.meshgrid(x,y,w,h)
# 	x,y,w,h = x.reshape((-1,1)), y.reshape((-1,1)), w.reshape((-1,1)), h.reshape((-1,1))
# 	boxes = np.concatenate((x,y,w,h), axis=1)

# 	boxes[:,0] = np.maximum(0., np.minimum(imsize[1], boxes[:,0]))
# 	boxes[:,1] = np.maximum(0., np.minimum(imsize[0], boxes[:,1]))
# 	idx = boxes[:,0] + boxes[:,2] > imsize[1]
# 	boxes[idx,2] = imsize[1] - boxes[idx,0]
# 	idx = boxes[:,1] + boxes[:,3] > imsize[0]
# 	boxes[idx,3] = imsize[0] - boxes[idx,1]

# 	return boxes

# def _generate_boxes(objs, imsize, num_pos=20, num_scale=20, mult=2):
# 	boxes = None
# 	for i in range(objs.__len__()):
# 		new_boxes = _generate_boxes_from_obj(objs[i], imsize, num_pos, num_scale, mult)
# 		if boxes is None:
# 			boxes = new_boxes
# 		else:
# 			boxes = np.concatenate((boxes, new_boxes), axis=0)
# 	return boxes

# def _boxes_as_annotations(boxes, obj=None):
# 	objs = []
# 	for i in range(boxes.shape[0]):
# 		box = boxes[i,:]
# 		new_obj = {}
# 		new_obj['xim'], new_obj['yim'], new_obj['wim'], new_obj['him'] = box[0], box[1], box[2], box[3]
# 		if obj is not None:
# 			new_obj['x'] = obj['x'] - new_obj['xim']
# 			new_obj['y'] = obj['y'] - new_obj['yim']
# 			new_obj['w'], new_obj['h'] = obj['w'], obj['h']
# 			new_obj['label'] = obj['label']
# 		else:
# 			new_obj['label'] = 'nothing'
# 		objs.append(new_obj)
# 	return objs

# def _calc_overlap(boxes, gt_boxes):
# 		xi = np.maximum(boxes[:,0], gt_boxes[:,0])
# 		yi = np.maximum(boxes[:,1], gt_boxes[:,1])
# 		xf = np.minimum(boxes[:,0] + boxes[:,2], gt_boxes[:,0] + gt_boxes[:,2])
# 		yf = np.minimum(boxes[:,1] + boxes[:,3], gt_boxes[:,1] + gt_boxes[:,3])
# 		w, h = np.maximum(0., xf - xi), np.maximum(0., yf - yi)
# 		isec = w * h
# 		boxes_size, gt_boxes_size = np.prod(boxes[:,2:], axis=1), np.prod(gt_boxes[:,2:], axis=1)
# 		union = boxes_size + gt_boxes_size - isec
# 		return isec / union, isec / gt_boxes_size

# def _find_valid_boxes(
# 		objs, boxes, 
# 		imsize,
# 		min_overlap=.5,
# 		max_obj_overlap=.3,
# 		min_neg_overlap=.1,
# 		min_obj_size=20*20,
# 		N=100, 
# 		neg=.75
# 	):
# 	neg_rois = int(np.ceil(neg * N))
# 	pos_rois = N - neg_rois
# #	 pos_rois_per_obj = int(np.ceil(float(pos_rois) / objs.__len__()))

# 	boxes = boxes.reshape(boxes.shape + (1,))

# 	gt_boxes = np.zeros((1, 4, objs.__len__()))
# 	for i in range(objs.__len__()):
# 		gt_boxes[0, :, i] = [objs[i]['x'], objs[i]['y'], objs[i]['w'], objs[i]['h']]

# 	iou, overlap = _calc_overlap(boxes, gt_boxes)

# 	max_overlap = overlap.max(axis=1)
# 	max_iou = iou.max(axis=1)
# 	max_iou_idx = iou.argmax(axis=1)
# 	sorted_overlap = np.sort(iou, axis=1)[:,::-1]
# 	idx = np.arange(boxes.shape[0])
# 	box_size = boxes[:,2:4,0].prod(axis=1)

# 	'''
# 	Calculate boxes whose max overlap is between .1 and .5, these constitute negative examples.
# 	'''
# 	neg_idx = np.bitwise_and(max_overlap > min_neg_overlap, max_overlap < min_overlap)
# 	neg_examples = boxes[np.random.choice(idx[neg_idx], size=neg_rois, replace=False)]

# 	valid_overlap_idx = max_iou > min_overlap
# 	big_enough_idx = box_size > min_obj_size
# 	if sorted_overlap.shape[1] > 1:
# 		distinct_box_idx = sorted_overlap[:,1] < max_obj_overlap
# 	else:
# 		distinct_box_idx = np.ones(neg_idx.shape, dtype=np.bool)

# 	'''
# 	Loop over each object in image and find rois which do not overlap with other objects too much
# 	while overlapping the object of interest a specified minimum amount.
# 	'''
# 	valid_example_idx = np.bitwise_and(
# 		np.bitwise_and(valid_overlap_idx, distinct_box_idx),
# 		big_enough_idx
# 	)

# 	if valid_example_idx.nonzero()[0].size > 0:
# 		pos_examples = np.concatenate((boxes[:,:,0], max_iou_idx.reshape((-1,1))), axis=1)
# 		pos_examples = pos_examples[np.random.choice(idx[valid_example_idx], size=pos_rois, replace=False)]
# 		return neg_examples, pos_examples
# 	else:
# 		return None, None

# def _data_from_annotation(annotation, size, num_classes, label2num, dtype=theano.config.floatX):
# 	imsize = annotation['size']
# 	objs = annotation['annotations']
	
# 	X = np.zeros((objs.__len__(),3) + size, dtype=dtype)
# 	labels = np.zeros((objs.__len__(), 4 + num_classes + 1))
	
# 	im = imread(annotation['image'])
# 	if im.dtype != dtype:
# 		im = im.astype(dtype)
# 	if im.shape.__len__() == 2:
# 		im = np.repeat(im.reshape(im.shape + (1,)), 3, axis=2)
# 	if im.shape[2] > 3:
# 		im = im[:,:,:3]
# 	if im.max() > 1:
# 		im /= 255
	
# 	# 	for i in range(objs.__len__()):
# 		obj = objs[i]
# 		gtruth = np.zeros(4 + num_classes + 1, dtype=dtype)
# 		gtruth[4 + label2num[obj['label']]] += 1
# 		xim, yim, wim, him = obj['xim'], obj['yim'], obj['wim'], obj['him']
# 		box = BoundingBox(xim, yim, xim + wim, yim + him)
# 		subim = box.subimage(im)

# 		# colour space augmentation
# 		subim = color.rgb2hsv(subim)
# 		subim[:,:,2] *= (0.6 * npr.rand() + 0.9)
# 		idx = subim[:,:,2] > 1.0
# 		subim[:,:,2][idx] = 1.0
# 		subim = color.hsv2rgb(subim)

# 		# randomly rotate
# 		flip_horz, flip_vert = npr.rand() > .5 ,npr.rand() > .5
# 		if flip_horz:
# 			subim = subim[:,::-1]
# 		if flip_vert:
# 			subim = subim[::-1,:]

# 		subim = resize(subim, size)

# 		if obj['label'] != 'nothing':
# 			x,y,w,h = obj['x'], obj['y'], obj['w'], obj['h']
# 			x_scale, y_scale = 1. / wim, 1. / him
# 			x, y, w, h = x * x_scale, y * y_scale, w * x_scale, h * y_scale

# 			# flip coordinates
# 			if flip_horz:
# 				x = 1 - (x + w)
# 			if flip_vert:
# 				y = 1 - (y + h)

# 			w, h = np.log(w), np.log(h)
# 			gtruth[:4] = [x, y, w, h]
# 		labels[i] = gtruth
# 		X[i] = subim.reshape((1,) + subim.shape).swapaxes(3,2).swapaxes(2,1)
# 	return X, labels

# def generate_rois(annotations, size, num_classes, label2num, num_batch=2, dtype=theano.config.floatX, N=100, neg=.75, min_obj_size=30*30):
# 	'''
# 	'''
# 	imsize = None
# 	np.random.shuffle(annotations)
# 	for i in tqdm(range(0,annotations.__len__(),num_batch)):
# 		X,y = None, None
# 		for j in range(min(annotations.__len__() - i, num_batch)):
# 			annotation = annotations[i + j]
# 			new_annotation = {}
# 			new_annotation['image'] = annotation['image']
# 			new_annotation['size'] = annotation['size']
# 			imsize = annotation['size']

# 			objs = annotation['annotations']
# 			new_objs = []

# 			boxes = _generate_boxes(objs, imsize, num_pos=20, num_scale=20, mult=2)

# 			neg_examples, pos_examples = _find_valid_boxes(
# 				objs,
# 				boxes,
# 				imsize,
# 				N=N,
# 				neg=neg,
# 				min_obj_size=min_obj_size
# 			)

# 			if neg_examples is not None:
# 				new_objs.extend(_boxes_as_annotations(neg_examples))
# 				for k in range(objs.__len__()):
# 					idx = np.equal(pos_examples[:,-1], k)
# 					if idx.nonzero() > 0:
# 						new_objs.extend(_boxes_as_annotations(pos_examples[idx,:4], objs[k]))
# 				new_annotation['annotations'] = new_objs

# 				Xim, yim = _data_from_annotation(
# 					new_annotation,
# 					size,
# 					num_classes,
# 					label2num
# 				)
# 				if X is None:
# 					X,y = Xim, yim
# 				else:
# 					X = np.concatenate((X,Xim), axis=0)
# 					y = np.concatenate((y,yim), axis=0)

# 		if X is not None:
# 			idx = np.arange(X.shape[0])
# 			np.random.shuffle(idx)
# 			yield X[idx].astype(dtype), y[idx].astype(dtype)


