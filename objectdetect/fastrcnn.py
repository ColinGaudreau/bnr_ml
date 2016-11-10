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

		# define normalization factor
		self.gamma = theano.shared(np.zeros((4,), dtype=theano.config.floatX), name='gamma', borrow=True)
		self.beta = theano.shared(np.ones((4,), dtype=theano.config.floatX), name='beta', borrow=True)

		self.params.extend([self.gamma, self.beta])

		# for detection
		self._trained = False

	def _get_cost(self, detection_output, localization_output, target, lmbda=1., eps=1e-4):
		'''
		detection_output: NxK
		localization_output: NxKx4
		'''
		# normalize input
		mu, var = T.mean(target[:,:4], axis=0), T.var(target[:,:4], axis=0)
		target = T.set_subtensor(
			target[:,:4], 
			((target[:,:4] - mu.dimshuffle('x',0)) / T.sqrt(var.dimshuffle('x',0) + eps)) * self.gamma.dimshuffle('x',0) + self.beta.dimshuffle('x',0)
		)

		class_idx = target[:,-(self.num_classes + 1):].argmax(axis=1)
		mask = T.ones((target.shape[0], 1))
		mask = T.switch(T.eq(target[:,-(self.num_classes + 1):].argmax(axis=1), self.num_classes), 0, 1) # mask for non-object ground truth labels

		cost = categorical_crossentropy(detection_output, target[:,-(self.num_classes + 1):])
		cost += lmbda * mask * T.sum(smooth_l1(localization_output[T.arange(localization_output.shape[0]), class_idx] - target[:,:4]), axis=1)
		
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
		train_fn = theano.function([self.input, target], cost, updates=updates)
		test_fn = theano.function([self.input, target], cost_test)
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
					err = train_fn(Xbatch, ybatch)
					train_loss_batch.append(err)
					print_obj.println('Batch error: %.4f' % err)
				
				for Xbatch, ybatch in test_gen:
					test_loss_batch.append(test_fn(Xbatch, ybatch))

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
		annotations = [[deepcopy(o) for o in obj] for obj in annotations]

		swap_axes = lambda im: im.swapaxes(2,1).swapaxes(1,0)

		for i in range(0,annotations.__len__(),per_batch):
			X, y = np.zeros((num_rios * per_batch, 3) + new_size), np.zeros((num_rios * per_batch, 4 + (num_classes + 1)))
			cnt = 0
			for j in range(min(per_batch, annotations.__len__() - i)):
				annotation = annotations[i+j]
				objs = annotation['annotations']
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
						pdb.set_trace()
						if np.prod(new_im.shape) > 0:
							X[cnt] = swap_axes(resize(new_im, new_size))
							y[cnt,:4], y[cnt,-(num_classes + 1):] = coord, label
							cnt += 1
			yield X[:cnt].astype(theano.config.floatX), y[:cnt].astype(theano.config.floatX)








































