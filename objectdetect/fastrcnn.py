import theano
from theano import tensor as T

import numpy as np

import lasagne
from lasagne.layers import get_output, get_all_params
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import rmsprop, sgd

from skimage.io import imread
from skimage.transform import resize

from bnr_ml.utils.nonlinearities import smooth_l1
from bnr_ml.objectdetect.utils import BoundingBox, transform_coord
from bnr_ml.utils.helpers import meshgrid2D

from copy import deepcopy
from itertools import tee
import time
import pdb
from tqdm import tqdm

class FastRCNNDetector(object):
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
		self._localize = reshape_loc_layer(get_output(network['localize'], deterministic=False), num_classes)
		self._localize_test = reshape_loc_layer(get_output(network['localize'], deterministic=True), num_classes)

		params, params_extra = get_all_params(network['detect']), get_all_params(network['localize'])
		for param in params_extra:
			if param not in params:
				params.append(param)
		self.params = params

		# for detection
		self._trained = False

	def _get_cost(self, detection_output, localization_output, target, lmbda=1., eps=1e-4):
		'''
		detection_output: NxK
		localization_output: NxKx4
		'''
		class_idx = target[:,-(self.num_classes + 1):].argmax(axis=1)
		mask = T.ones((target.shape[0], 1))
		mask = T.switch(T.eq(target[:,-(self.num_classes + 1):].argmax(axis=1), self.num_classes), 0, 1) # mask for non-object ground truth labels

		cost = categorical_crossentropy(detection_output, target[:,-(self.num_classes + 1):])
		cost += lmbda * mask * T.sum(smooth_l1(localization_output[T.arange(localization_output.shape[0]), class_idx] - target[:,:4]), axis=1)
		pdb.set_trace()
		return T.mean(cost)

	def train(
			self,
			train_gen,
			test_gen,
			print_obj,
			updates=rmsprop,
			epochs=10,
			lr=1e-4,
			momentum=0.9,
			lmbda=1.,
		):
		'''
		'''	
		self._trained = True
		target = T.matrix('target')

		print_obj.println('Getting cost...')
		cost = self._get_cost(self._detect, self._localize, target, lmbda=lmbda)
		if test_gen is not None:
			cost_test = self._get_cost(self._detect_test, self._localize_test, target, lmbda=lmbda)

		updates = rmsprop(cost, self.params, learning_rate=lr)
		
		print_obj.println('Compiling...')
		ti = time.time(); time.sleep(.1)
		self._train_fn = theano.function([self.input, target], cost, updates=updates)
		self._test_fn = theano.function([self.input, target], cost_test)
		print_obj.println('Compiling took %.3f seconds' % (time.time() - ti,))

		train_loss, test_loss = np.zeros(epochs), np.zeros(epochs)

		print_obj.println('Beginning training')

		try:
			for epoch in range(epochs):
				train_loss_batch = []
				test_loss_batch = []

				train_gen, train_gen_backup = tee(train_gen)
				test_gen, test_gen_backup = tee(test_gen)
				
				ti = time.time()
				for Xbatch, ybatch in train_gen:
					err = self._train_fn(Xbatch, ybatch)
					train_loss_batch.append(err)
					print_obj.println('Batch error: %.4f' % err)
				
				for Xbatch, ybatch in test_gen:
					test_loss_batch.append(self._test_fn(Xbatch, ybatch))

				train_loss[epoch] = np.mean(train_loss_batch)
				test_loss[epoch] = np.mean(test_loss_batch)

				train_gen, test_gen = train_gen_backup, test_gen_backup

				print_obj.println('\nEpoch %d\n--------\nTrain Loss: %.4f, Test Loss: %.4f' % \
					(epoch, train_loss[epoch], test_loss[epoch]))
				print_obj.println('Epoch took %.3f seconds.' % (time.time() - ti,))
				time.sleep(.05)
		except KeyboardInterrupt:
			pass

		return train_loss[train_loss > 0], test_loss[test_loss > 0]

	def detect(self, im, proposals=None, thresh=.7):
		if im.shape.__len__() == 2:
			im = np.repeat(im.reshape(im.shape + (1,)), 3, axis=2)
		if im.shape[2] > 3:
			im = im[:,:,:3]
		if im.max() > 1:
			im = im / 255.
		if im.dtype != theano.config.floatX:
			im = im.astype(theano.config.floatX)

		if self._trained or not hasattr(self, '_detect_fn'):
			self._detect_fn = theano.function([self.input], [self._detect_test, self._localize_test])
			self._trained = False

		swap = lambda im: im.reshape((1,) + im.shape).swapaxes(3,2).swapaxes(2,1).astype(theano.config.floatX)

		if proposals is not None:
			ims = np.zeros((proposals.shape[0],3) + self.input_shape, dtype=theano.config.floatX)
			cnt = 0
			regions = []
			for i in range(proposals.shape[0]):
				box = BoundingBox(
					proposals[i, 0],
					proposals[i, 1],
					proposals[i, 0] + proposals[i, 2],
					proposals[i, 1] + proposals[i, 3]
				)
				if box.size > 0:
					subim = box.subimage(im)
					if np.prod(im.shape) > 0:
						subim = resize(subim, self.input_shape)
						ims[cnt] = swap(subim)
						regions.append(box)
						cnt += 1
			region = np.asarray(regions)
			class_score, coord = self._detect_fn(ims[:cnt])
			class_idx, obj_idx = np.argmax(class_score, axis=1), np.max(class_score[:,:-1], axis=1) > thresh
			coord = coord[np.arange(coord.shape[0]), class_idx]
			coord[:,[2,3]] = np.exp(coord[:,[2,3]])
			class_score, coord = class_score[obj_idx], coord[obj_idx]
			for i in range(coord.shape[0]):
				coord[i,[0,2]] *= regions[i].w
				coord[i,[1,3]] *= regions[i].h
				coord[i,0] += regions[i].xi
				coord[i,1] += regions[i].yi
				coord[i,[0,2]] /= im.shape[1]
				coord[i,[1,3]] /= im.shape[0]
			return class_score, coord
		else:
			im = resize(im, self.input_shape + (3,))

			preds = self._detect_fn(swap(im).astype(theano.config.floatX))
			class_score, coord = preds[0][0], preds[1][0]
			
			cls_idx = np.argmax(class_score)
			coord = coord[cls_idx]
			coord[[2,3]] = np.exp(coord[[2,3]])
			return class_score, coord
		
	@staticmethod
	def generate_data(
			annotations,
			new_size,
			num_classes,
			per_batch=2,
			num_rios=50,
			min_overlap=.5
		):

		swap_axes = lambda im: im.swapaxes(2,1).swapaxes(1,0)

		for i in range(0,annotations.__len__(),per_batch):
			X, y = np.zeros((num_rios * per_batch, 3) + new_size), np.zeros((num_rios * per_batch, 4 + (num_classes + 1)))
			cnt = 0
			for j in range(min(per_batch, annotations.__len__() - i)):
				annotation = annotations[i+j]
				objs = [deepcopy(o) for o in annotation['annotations']]
				im = imread(annotation['image'])

				if im.shape.__len__() == 2:
					im = np.repeat(im.reshape(im.shape + (1,)), 3, axis=2)
				elif im.shape[2] > 3:
					im = im[:,:,:3]
				if im.max() > 1:
					im = im / 255.
				if im.dtype != theano.config.floatX:
					im = im.astype(theano.config.floatX)

				for k in range(num_rios):
					coord, label = np.zeros(4), np.zeros(num_classes + 1)
					obj = objs[int(objs.__len__() * np.random.rand())]
					to_be_localized = np.random.rand() < .5

					if to_be_localized:
						iou = 0.7 + 0.3 * np.random.rand()
						label[obj['label']] = 1.
					else:
						iou = 0.1 + 0.2 * np.random.rand()
						label[num_classes] = 1.
					obj_box = BoundingBox(
						obj['x'],
						obj['y'],
						obj['x'] + obj['w'],
						obj['y'] + obj['h']
					)
					new_box = BoundingBox.gen_randombox(iou, obj_box)
					if new_box.isvalid():
						if to_be_localized:
							coord[0] = (obj_box.xi - new_box.xi) / (new_box.w)
							coord[1] = (obj_box.yi - new_box.yi) / (new_box.h)
							coord[2] = 100 * np.log(obj_box.w / new_box.w)
							coord[3] = 100 * np.log(obj_box.h / new_box.h)
						new_im = new_box.subimage(im)
						if np.prod(new_im.shape) > 0:
							X[cnt] = swap_axes(resize(new_im, new_size))
							y[cnt,:4], y[cnt,-(num_classes + 1):] = coord, label
							cnt += 1
			yield X[:cnt].astype(theano.config.floatX), y[:cnt].astype(theano.config.floatX)

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
	boxes[idx,2] = imsize[1]
	idx = boxes[:,1] + boxes[:,3] > imsize[0]
	boxes[idx,3] = imsize[0]

	return boxes


# In[957]:

def _generate_boxes(objs, imsize, num_pos=20, num_scale=20, mult=2):
	boxes = None
	for i in range(objs.__len__()):
		new_boxes = _generate_boxes_from_obj(objs[i], imsize, num_pos, num_scale, mult)
		if boxes is None:
			boxes = new_boxes
		else:
			boxes = np.concatenate((boxes, new_boxes), axis=0)
	return boxes


# In[915]:

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


# In[916]:

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


# In[937]:

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

	pos_examples = np.concatenate((boxes[:,:,0], max_iou_idx.reshape((-1,1))), axis=1)
	pos_examples = pos_examples[np.random.choice(idx[valid_example_idx], size=pos_rois, replace=False)]
	return neg_examples, pos_examples

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
			x, y, w, h = x * xscale, y * yscale, w * xscale, h * yscale
			gtruth[:4] = [x, y, w, h]
		labels[i] = gtruth
		X[i] = subim.reshape((1,) + subim.shape).swapaxes(3,2).swapaxes(2,1)
	return X, labels


# In[1044]:

def generate_rois(annotations, size, num_classes, label2num, num_batch=2, dtype=theano.config.floatX, N=100, neg=.75, min_obj_size=30*30):
	'''
	'''
	imsize = None
	np.random.shuffle(annotations)
	for i in tqdm(range(0,annotations.__len__(),2)):
		X,y = None, None
		for i in range(min(annotations.__len__() - i, 2)):
			annotation = annotations[i]
	#		 if imsize is not None:
	#			 boxes[:,[0,2]] /= imsize[1]
	#			 boxes[:,[1,3]] /= imsize[0] 

			new_annotation = {}
			new_annotation['image'] = annotation['image']
			new_annotation['size'] = annotation['size']
			imsize = annotation['size']

	#		 boxes[:,[0,2]] *= imsize[1]
	#		 boxes[:,[1,3]] *= imsize[0]

			objs = annotation['annotations']
			new_objs = []

			boxes = _generate_boxes(objs, imsize, num_pos=20, num_scale=10, mult=2)
	#		 boxes = _gen_boxes(imsize, num_pos=20, num_scale=20)

			neg_examples, pos_examples = _find_valid_boxes(
				objs,
				boxes,
				imsize,
				N=N, 
				neg=neg,
				min_obj_size=min_obj_size
			)

			new_objs.extend(_boxes_as_annotations(neg_examples))
			for i in range(objs.__len__()):
				idx = np.equal(pos_examples[:,-1], i)
				if idx.nonzero() > 0:
					new_objs.extend(_boxes_as_annotations(pos_examples[idx,:4], objs[i]))
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
		idx = np.arange(X.shape[0])
		np.random.shuffle(idx)
		pdb.set_trace()
		yield X[idx].astype(dtype), y[idx].astype(dtype)

