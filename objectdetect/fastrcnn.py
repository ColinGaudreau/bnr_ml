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
import cv2

from bnr_ml.utils.nonlinearities import smooth_l1
from bnr_ml.objectdetect.utils import BoundingBox
from bnr_ml.utils.helpers import meshgrid2D
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
		if self._trained or not hasattr(self, '_detect_fn'):
			self._detect_input_ndarray = np.zeros((batch_size,3) + self.input_shape, dtype=theano.config.floatX)
			self._detect_input = theano.shared(self._detect_input_ndarray, name='detection_input', borrow=True)

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

			self._detect_input_ndarray[batch_index] = swap(subim_ph)
			batch_index += 1

			if batch_index == batch_size:
				self._detect_input.set_value(self._detect_input_ndarray, borrow=True)
				class_score[i - (batch_size - 1):i + 1], coord[i - (batch_size - 1):i + 1] = self._detect_fn()
				batch_index = 0

		if batch_index != batch_size and batch_index != 0:
			self._detect_input.set_value(self._detect_input_ndarray[:batch_index], borrow=True)
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
			boxes, scores = nms(objects[cls_idx], scores=class_score[cls_idx], overlap=overlap)
			cls_output['boxes'] = boxes
			cls_output['scores'] = scores
			if num_to_label is not None:
				cls = num_to_label[cls]
			output[cls] = cls_output

		return output
		
	# @staticmethod
	# def generate_data(
	# 		annotations,
	# 		new_size,
	# 		num_classes,
	# 		per_batch=2,
	# 		num_rios=50,
	# 		min_overlap=.5
	# 	):

	# 	swap_axes = lambda im: im.swapaxes(2,1).swapaxes(1,0)

	# 	for i in range(0,annotations.__len__(),per_batch):
	# 		X, y = np.zeros((num_rios * per_batch, 3) + new_size), np.zeros((num_rios * per_batch, 4 + (num_classes + 1)))
	# 		cnt = 0
	# 		for j in range(min(per_batch, annotations.__len__() - i)):
	# 			annotation = annotations[i+j]
	# 			objs = [deepcopy(o) for o in annotation['annotations']]
	# 			im = imread(annotation['image'])

	# 			if im.shape.__len__() == 2:
	# 				im = np.repeat(im.reshape(im.shape + (1,)), 3, axis=2)
	# 			elif im.shape[2] > 3:
	# 				im = im[:,:,:3]
	# 			if im.max() > 1:
	# 				im = im / 255.
	# 			if im.dtype != theano.config.floatX:
	# 				im = im.astype(theano.config.floatX)

	# 			for k in range(num_rios):
	# 				coord, label = np.zeros(4), np.zeros(num_classes + 1)
	# 				obj = objs[int(objs.__len__() * np.random.rand())]
	# 				to_be_localized = np.random.rand() < .5

	# 				if to_be_localized:
	# 					iou = 0.7 + 0.3 * np.random.rand()
	# 					label[obj['label']] = 1.
	# 				else:
	# 					iou = 0.1 + 0.2 * np.random.rand()
	# 					label[num_classes] = 1.
	# 				obj_box = BoundingBox(
	# 					obj['x'],
	# 					obj['y'],
	# 					obj['x'] + obj['w'],
	# 					obj['y'] + obj['h']
	# 				)
	# 				new_box = BoundingBox.gen_randombox(iou, obj_box)
	# 				if new_box.isvalid():
	# 					if to_be_localized:
	# 						coord[0] = (obj_box.xi - new_box.xi) / (new_box.w)
	# 						coord[1] = (obj_box.yi - new_box.yi) / (new_box.h)
	# 						coord[2] = 100 * np.log(obj_box.w / new_box.w)
	# 						coord[3] = 100 * np.log(obj_box.h / new_box.h)
	# 					new_im = new_box.subimage(im)
	# 					if np.prod(new_im.shape) > 0:
	# 						X[cnt] = swap_axes(resize(new_im, new_size))
	# 						y[cnt,:4], y[cnt,-(num_classes + 1):] = coord, label
	# 						cnt += 1
	# 		yield X[:cnt].astype(theano.config.floatX), y[:cnt].astype(theano.config.floatX)

def _gen_boxes(imsize, num_pos=20, num_scale=20):
	x = np.linspace(0, imsize[1], num_pos)
	y = np.linspace(0, imsize[0], num_pos)
	w = np.linspace(0, imsize[1], num_scale)
	h = np.linspace(0, imsize[0], num_scale)
	
	x,y,w,h = np.meshgrid(x,y,w,h)
	x,y,w,h = x.reshape((-1,1)), y.reshape((-1,1)), w.reshape((-1,1)), h.reshape((-1,1))
	boxes = np.concatenate((x,y,w,h), axis=1)

	boxes[:,0] = np.maximum(0., np.minimum(imsize[0], boxes[:,0]))
	boxes[:,1] = np.maximum(0., np.minimum(1, boxes[:,1]))
	idx = boxes[:,0] + boxes[:,2] > imsize[1]
	boxes[idx,2] = imsize[1]
	idx = boxes[:,1] + boxes[:,3] > imsize[0]
	boxes[idx,3] = imsize[0]

	return boxes

def _generate_boxes_from_obj(obj, imsize, num_pos=20, num_scale=20, mult=2):
	x = np.linspace(max(0,obj['x'] - .7*obj['w']), min(imsize[1], obj['x'] + .7*obj['w']), num_pos)
	y = np.linspace(max(0,obj['y'] - .7*obj['h']), min(imsize[0], obj['y'] + .7*obj['h']), num_pos)
	w = np.linspace(obj['w'] / mult, min(imsize[1], obj['w'] * mult), num_pos)
	h = np.linspace(obj['h'] / mult, min(imsize[0], obj['h'] * mult), num_pos)
		
	x,y,w,h = np.meshgrid(x,y,w,h)
	x,y,w,h = x.reshape((-1,1)), y.reshape((-1,1)), w.reshape((-1,1)), h.reshape((-1,1))
	boxes = np.concatenate((x,y,w,h), axis=1)

	boxes[:,0] = np.maximum(0., np.minimum(imsize[1], boxes[:,0]))
	boxes[:,1] = np.maximum(0., np.minimum(imsize[0], boxes[:,1]))
	idx = boxes[:,0] + boxes[:,2] > imsize[1]
	boxes[idx,2] = imsize[1] - boxes[idx,0]
	idx = boxes[:,1] + boxes[:,3] > imsize[0]
	boxes[idx,3] = imsize[0] - boxes[idx,1]

	return boxes

def _generate_boxes(objs, imsize, num_pos=20, num_scale=20, mult=2):
	boxes = None
	for i in range(objs.__len__()):
		new_boxes = _generate_boxes_from_obj(objs[i], imsize, num_pos, num_scale, mult)
		if boxes is None:
			boxes = new_boxes
		else:
			boxes = np.concatenate((boxes, new_boxes), axis=0)
	return boxes

def _boxes_as_annotations(boxes, obj=None):
	objs = []
	for i in range(boxes.shape[0]):
		box = boxes[i,:]
		new_obj = {}
		new_obj['xim'], new_obj['yim'], new_obj['wim'], new_obj['him'] = box[0], box[1], box[2], box[3]
		if obj is not None:
			new_obj['x'] = obj['x'] - new_obj['xim']
			new_obj['y'] = obj['y'] - new_obj['yim']
			new_obj['w'], new_obj['h'] = obj['w'], obj['h']
			new_obj['label'] = obj['label']
		else:
			new_obj['label'] = 'nothing'
		objs.append(new_obj)
	return objs

def _calc_overlap(boxes, gt_boxes):
		xi = np.maximum(boxes[:,0], gt_boxes[:,0])
		yi = np.maximum(boxes[:,1], gt_boxes[:,1])
		xf = np.minimum(boxes[:,0] + boxes[:,2], gt_boxes[:,0] + gt_boxes[:,2])
		yf = np.minimum(boxes[:,1] + boxes[:,3], gt_boxes[:,1] + gt_boxes[:,3])
		w, h = np.maximum(0., xf - xi), np.maximum(0., yf - yi)
		isec = w * h
		boxes_size, gt_boxes_size = np.prod(boxes[:,2:], axis=1), np.prod(gt_boxes[:,2:], axis=1)
		union = boxes_size + gt_boxes_size - isec
		return isec / union, isec / gt_boxes_size

def _find_valid_boxes(
		objs, boxes, 
		imsize,
		min_overlap=.5,
		max_obj_overlap=.3,
		min_neg_overlap=.1,
		min_obj_size=20*20,
		N=100, 
		neg=.75
	):
	neg_rois = int(np.ceil(neg * N))
	pos_rois = N - neg_rois
#	 pos_rois_per_obj = int(np.ceil(float(pos_rois) / objs.__len__()))

	boxes = boxes.reshape(boxes.shape + (1,))

	gt_boxes = np.zeros((1, 4, objs.__len__()))
	for i in range(objs.__len__()):
		gt_boxes[0, :, i] = [objs[i]['x'], objs[i]['y'], objs[i]['w'], objs[i]['h']]

	iou, overlap = _calc_overlap(boxes, gt_boxes)

	max_overlap = overlap.max(axis=1)
	max_iou = iou.max(axis=1)
	max_iou_idx = iou.argmax(axis=1)
	sorted_overlap = np.sort(iou, axis=1)[:,::-1]
	idx = np.arange(boxes.shape[0])
	box_size = boxes[:,2:4,0].prod(axis=1)

	'''
	Calculate boxes whose max overlap is between .1 and .5, these constitute negative examples.
	'''
	neg_idx = np.bitwise_and(max_overlap > min_neg_overlap, max_overlap < min_overlap)
	neg_examples = boxes[np.random.choice(idx[neg_idx], size=neg_rois, replace=False)]

	valid_overlap_idx = max_iou > min_overlap
	big_enough_idx = box_size > min_obj_size
	if sorted_overlap.shape[1] > 1:
		distinct_box_idx = sorted_overlap[:,1] < max_obj_overlap
	else:
		distinct_box_idx = np.ones(neg_idx.shape, dtype=np.bool)

	'''
	Loop over each object in image and find rois which do not overlap with other objects too much
	while overlapping the object of interest a specified minimum amount.
	'''
	valid_example_idx = np.bitwise_and(
		np.bitwise_and(valid_overlap_idx, distinct_box_idx),
		big_enough_idx
	)

	if valid_example_idx.nonzero()[0].size > 0:
		pos_examples = np.concatenate((boxes[:,:,0], max_iou_idx.reshape((-1,1))), axis=1)
		pos_examples = pos_examples[np.random.choice(idx[valid_example_idx], size=pos_rois, replace=False)]
		return neg_examples, pos_examples
	else:
		return None, None

def _data_from_annotation(annotation, size, num_classes, label2num, dtype=theano.config.floatX):
	imsize = annotation['size']
	objs = annotation['annotations']
	
	X = np.zeros((objs.__len__(),3) + size, dtype=dtype)
	labels = np.zeros((objs.__len__(), 4 + num_classes + 1))
	
	im = imread(annotation['image'])
	if im.dtype != dtype:
		im = im.astype(dtype)
	if im.shape.__len__() == 2:
		im = np.repeat(im.reshape(im.shape + (1,)), 3, axis=2)
	if im.shape[2] > 3:
		im = im[:,:,:3]
	if im.max() > 1:
		im /= 255
	
	for i in range(objs.__len__()):
		obj = objs[i]
		gtruth = np.zeros(4 + num_classes + 1, dtype=dtype)
		gtruth[4 + label2num[obj['label']]] += 1
		xim, yim, wim, him = obj['xim'], obj['yim'], obj['wim'], obj['him']
		box = BoundingBox(xim, yim, xim + wim, yim + him)
		subim = box.subimage(im)
		subim = resize(subim, size)
		if obj['label'] != 'nothing':
			x,y,w,h = obj['x'], obj['y'], obj['w'], obj['h']
			xscale, yscale = 1. / wim, 1. / him
			x, y, w, h = x * xscale, y * yscale, np.log(w * xscale), np.log(h * yscale)
			gtruth[:4] = [x, y, w, h]
		labels[i] = gtruth
		X[i] = subim.reshape((1,) + subim.shape).swapaxes(3,2).swapaxes(2,1)
	return X, labels

def generate_rois(annotations, size, num_classes, label2num, num_batch=2, dtype=theano.config.floatX, N=100, neg=.75, min_obj_size=30*30):
	'''
	'''
	imsize = None
	np.random.shuffle(annotations)
	for i in tqdm(range(0,annotations.__len__(),num_batch)):
		X,y = None, None
		for j in range(min(annotations.__len__() - i, num_batch)):
			annotation = annotations[i + j]
			new_annotation = {}
			new_annotation['image'] = annotation['image']
			new_annotation['size'] = annotation['size']
			imsize = annotation['size']

			objs = annotation['annotations']
			new_objs = []

			boxes = _generate_boxes(objs, imsize, num_pos=20, num_scale=20, mult=2)

			neg_examples, pos_examples = _find_valid_boxes(
				objs,
				boxes,
				imsize,
				N=N, 
				neg=neg,
				min_obj_size=min_obj_size
			)

			if neg_examples is not None:
				new_objs.extend(_boxes_as_annotations(neg_examples))
				for k in range(objs.__len__()):
					idx = np.equal(pos_examples[:,-1], k)
					if idx.nonzero() > 0:
						new_objs.extend(_boxes_as_annotations(pos_examples[idx,:4], objs[k]))
				new_annotation['annotations'] = new_objs

				Xim, yim = _data_from_annotation(
					new_annotation,
					size,
					num_classes,
					label2num
				)
				if X is None:
					X,y = Xim, yim
				else:
					X = np.concatenate((X,Xim), axis=0)
					y = np.concatenate((y,yim), axis=0)

		if X is not None:
			idx = np.arange(X.shape[0])
			np.random.shuffle(idx)
			yield X[idx].astype(dtype), y[idx].astype(dtype)

