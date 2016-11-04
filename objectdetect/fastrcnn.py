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

		def reshape_loc_layer(loc_layer, num_classes):
			return loc_layer.reshape((-1, num_classes + 1, 4))

		self.detect = get_output(network['detect'], deterministic=False)
		self.detect_test = get_output(network['detect'], deterministic=True)
		self.localize = reshape_loc_layer(get_output(network['localize'], deterministic=False), num_classes)
		self.localize_test = reshape_loc_layer(get_output(network['localize'], deterministic=True), num_classes)

		params, params_extra = get_all_params(network['detect']), get_all_params(network['localize'])
		for param in params_extra:
			if param not in params:
				params.append(param)
		self.params = params

	def _get_cost(self, detection_output, localization_output, target, lmbda=1.):
		'''
		detection_output: NxK
		localization_output: NxKx4
		'''

		class_idx = target[:,-(self.num_classes + 1):].argmax(axis=1, keepdims=True)
		mask = T.ones((target.shape[0], 1))
		mask = T.switch(T.eq(target[:,-(self.num_classes + 1):].argmax(axis=1), self.num_classes), 0, 1) # mask for non-object ground truth labels

		corr_loc, _ = meshgrid2D(T.arange(localization_output.shape[1]), T.arange(localization_output.shape[0]))
		corr_loc = T.eq(corr_loc, class_idx)
		corr_loc = T.repeat(corr_loc, 4, axis=2)

		pdb.set_trace()

		cost = categorical_crossentropy(detection_output, target[:,-(self.num_classes + 1):])
		cost += lmbda * mask * T.sum(smooth_l1(localization_output[corr_loc.nonzero()] - target[:,:4]), axis=1)

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
		target = T.matrix('target')

		print_obj.println('Getting cost...')
		cost = self._get_cost(self.detect, self.localize, target, lmbda=lmbda)
		if test_gen is not None:
			cost_test = self._get_cost(self.detect_test, self.localize_test, target, lmbda=lmbda)

		updates = rmsprop(cost, self.params)

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
				test_gen, test_gen_backup = tee(train_gen)

				for Xbatch, ybatch in train_gen:
					pdb.set_trace()
					err = train_fn(Xbatch, ybatch)
					train_loss_batch.append(err)
					print_obj.println('Batch error: %.4f' % err)

				for Xbatch, ybatch in test_gen:
					test_loss_batch.append(train_fn(Xbatch, ybatch))

				train_loss[epoch] = np.mean(train_loss_batch)
				test_loss[epoch] = np.mean(test_loss_batch)

				train_gen, test_gen = train_gen_backup, test_gen_backup

				print_obj.println('\nEpoch %d\n--------\nTrain Loss: %.4f, Test Loss: %.4f' % \
					(epoch, train_loss[epoch], test_loss[epoch]))
				time.sleep(.05)
		except KeyboardInterrupt:
			pass

		return train_loss[train_loss > 0], test_loss[test_loss > 0]


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

		for i in range(0,annotations.__len__(),2):
			X, y = np.zeros((num_rios * per_batch, 3) + new_size), np.zeros((num_rios * per_batch, 4 + (num_classes + 1)))
			cnt = 0
			for j in range(per_batch):
				annotation = annotations[i+j]
				im = imread(annotation[0]['image'])

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
					obj = annotation[int(annotation.__len__() * np.random.rand())]
					to_be_localized = np.random.rand() < .25

					if to_be_localized:
						iou = 0.5 + 0.5 *np.random.rand()
						label[obj['label']] = 1.
					else:
						iou = 0.1 + 0.4 * np.random.rand()
						label[num_classes] = 1.
					obj_box = BoundingBox(
						obj['x'],
						obj['y'],
						obj['w'],
						obj['h']
					)
					new_box = BoundingBox.gen_randombox(iou, obj_box)
					if to_be_localized:
						coord[0] = (new_box.xi - obj_box.xi) / (new_box.w)
						coord[1] = (new_box.yi - obj_box.yi) / (new_box.h)
						coord[2] = np.log(obj_box.w / new_box.w)
						coord[3] = np.log(obj_box.h / new_box.h)

					new_im = new_box.subimage(im)
					X[cnt] = swap_axes(resize(new_im, new_size))
					y[cnt,:4], y[cnt,-(num_classes + 1):] = coord, label
					cnt += 1
			yield X[:cnt], y[:cnt]








































