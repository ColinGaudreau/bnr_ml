import theano
from theano import tensor as T

import numpy as np
import numpy.random as npr

import lasagne
from lasagne.layers import get_output, get_all_params, Layer
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
from bnr_ml.logger.learning_objects import BaseLearningObject, BaseLearningSettings

from bnr_ml.objectdetect.fastrcnn.roi_layer import ROILayer
import bnr_ml.objectdetect.fastrcnn.datagen as datagen

from copy import deepcopy
from itertools import tee
import time
import pdb
from tqdm import tqdm

import dlib

class FastRCNNSettings(BaseLearningSettings):
	def __init__(
			self,
			train_annotations,
			test_annotations,
			train_args,
			test_args,
			data_generator=datagen.generate_data,
			print_obj=StreamPrinter(open('/dev/stdout','w')),
			update_fn=rmsprop,
			update_args={'learning_rate': 1e-5},
			lmbda=1.,
			hyperparameters={}
		):
		super(FastRCNNSettings, self).__init__()
		self.train_annotations = train_annotations
		self.test_annotations = test_annotations
		self.train_args = train_args
		self.test_args = test_args
		self.data_generator = data_generator
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
			roi_layer: ROI layer
			localize: FC Layer for localization

	'''
	def __init__(
		self,
		network,
		num_classes,
	):
		assert('detect' in network)
		assert('localize' in network)
		assert('roi_layer' in network)
		# assert(isinstance(network['roi_layer'], ROILayer))

		self.network = network
		self.num_classes = num_classes
		self.input = network['input'].input_var
		self.input_shape = network['input'].shape[2:]
		self.boxes = network['roi_layer'].boxes

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
		
		cost = T.sum(-target[:,-(self.num_classes + 1):] * T.log(detection_output), axis=1).mean()
		
		# cost = categorical_crossentropy(detection_output, target[:,-(self.num_classes + 1):])
		if lmbda > 0:
			cost += lmbda * T.sum(mask * T.sum(smooth_l1(localization_output[T.arange(localization_output.shape[0]), class_idx] - target[:,:4]), axis=1)) / mask.nonzero()[0].size
		
		return cost

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
		test_args = self.settings.test_args
		data_generator = self.settings.data_generator
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
			self._train_fn = theano.function([self.input, self.boxes, target], cost, updates=updates)
			print_obj.println('Compiling training function took %.3f seconds' % (time.time() - ti,))
			ti = time.time();
			self._test_fn = theano.function([self.input, self.boxes, target], cost_test)
			print_obj.println('Compiling test function took %.3f seconds' % (time.time() - ti,))

		print_obj.println('Beginning training')

		train_loss_batch = []
		test_loss_batch = []
		
		ti = time.time()
		for Xbatch, boxes_batch, ybatch in data_generator(train_annotations, **train_args):
			err = self._train_fn(Xbatch, boxes_batch, ybatch)
			train_loss_batch.append(err)
			print_obj.println('Batch error: %.4f' % err)
		
		for Xbatch, boxes_batch, ybatch in data_generator(test_annotations, **test_args):
			test_loss_batch.append(self._test_fn(Xbatch, boxes_batch, ybatch))

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
			n_apply=1,
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
		im_size = im.shape[:2]

		# compile detection function if it has not yet been done
		detect_input_ndarray = np.zeros((batch_size,3) + self.input_shape, dtype=theano.config.floatX)
		if self._trained or not hasattr(self, '_detect_fn'):
			self._detect_fn = theano.function([self.input, self.boxes], [self._detect_test, self._localize_test])
			self._trained = False
		
		swap = lambda im: im.swapaxes(2,1).swapaxes(1,0)
		regions = np.asarray(self._filter_regions(self._propose_regions(im, kvals, min_size), min_w, min_h))
		if max_regions is not None:
			max_regions = min(regions.shape[0], max_regions)
			regions = regions[npr.choice(regions.shape[0], max_regions, replace=False)]
		boxes = np.asarray([b.tolist() for b in regions], dtype=theano.config.floatX)
		boxes[:,[0,2]] /= im_size[1]; boxes[:,[1,3]] /= im_size[0]
		boxes = boxes.reshape((1,) + boxes.shape)
		
		im = resize(im, self.input_shape)
                im = swap(im).reshape((1,3) + im.shape[:2]).astype(theano.config.floatX)

		class_score, coord = self._detect_fn(im, boxes)

		# filter out windows which are 1) not labeled to be an object 2) below threshold
		class_id = np.argmax(class_score[:,:-1], axis=1)
		class_score = class_score[np.arange(class_score.shape[0]), class_id]
		is_obj = class_score > thresh
		coord = coord[np.arange(coord.shape[0]), class_id]
		coord[:,2:] = np.exp(coord[:,2:])
		coord[:,:2] -= coord[:,2:]/2

		# re-adjust boxes for the image
		objects = []
		scale_factor = (float(old_size[0])/im_size[0], float(old_size[1])/im_size[1])
		for i, box in enumerate(regions):
			# check if object confidence above threshold
			if is_obj[i]:
				# adjust box coordinates relative to the proposal region and original image size
				coord[i, [0,2]] *= box.w
				coord[i, [1,3]] *= box.h
				coord[i, 0] += box.xi
				coord[i, 1] += box.yi
				coord[i, 2:] += coord[i, :2]
				cls = class_id[i]
				if num_to_label is not None:
					cls = num_to_label[cls]
				obj = BoundingBox(*coord[i,:].tolist(), cls=cls, confidence=class_score[i])
				obj *= scale_factor
				objects.append(obj)
		
		# do nms
		objects = nms(objects, overlap=overlap, n_apply=n_apply)
		
		return objects


