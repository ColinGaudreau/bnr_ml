import theano
from theano import tensor as T
import numpy as np

from bnr_ml.nnet.updates import momentum as momentum_update
from bnr_ml.nnet.layers import AbstractNNetLayer
from bnr_ml.utils.helpers import meshgrid2D, bitwise_not, StreamPrinter
from bnr_ml.utils.nonlinearities import softmax, smooth_l1, smooth_abs, safe_sqrt
from bnr_ml.objectdetect import utils
from bnr_ml.objectdetect.nms import nms

from collections import OrderedDict
from tqdm import tqdm
import time
from PIL import Image, ImageDraw

from lasagne import layers
from lasagne.updates import rmsprop, sgd, adam
from lasagne.updates import momentum as momentum_update

from itertools import tee

from ml_logger.learning_objects import BaseLearningObject, BaseLearningSettings

import cv2

import pdb

class Yolo2Settings(BaseLearningSettings):
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
			lambda_obj=5.,
			lambda_noobj=1.,
			lambda_noobj_coord=1.,
			rescore=True,
			hyperparameters={}
		):
		super(Yolo2Settings, self).__init__()
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
		self.lambda_obj = lambda_obj
		self.lambda_noobj = lambda_noobj
		self.lambda_noobj_coord = lambda_noobj_coord
		self.rescore = rescore
		self.hyperparameters = {}

	def serialize(self):
		serialization = {}
		serialization['update_fn'] = self.update_fn.__str__()
		serialization['update_args'] = self.update_args
		serialization['lambda_obj'] = self.lambda_obj
		serialization['lambda_noobj'] = self.lambda_noobj
		serialization['lambda_noobj_coord'] = self.lambda_noobj_coord
		serialization['rescore'] = self.rescore  
		serialization.update(self.hyperparameters)
		return serialization

class Yolo2ObjectDetector(object):
	def __init__(
		self,
		network,
		num_classes,
		boxes=[(5.,.5), (.25,.5), (.5,.25), (.3,.3)]
	):
		assert('detection' in network and 'input' in network)
		super(Yolo2ObjectDetector, self).__init__()	
		self.network = network
		self.num_classes = num_classes
		self.boxes = boxes
		
		self.input = network['input'].input_var
		self.output_shape = layers.get_output_shape(network['detection'])[-2:]
		self.input_shape = network['input'].shape[-2:]
#		 self._hyperparameters = {'ratios': ratios, 'smin': smin, 'smax': smax}

		# set objectness predictor nonlinearity to sigmoid and
		# class predictor to softmax
		output = T.reshape(layers.get_output(network['detection']), (-1,self.boxes.__len__(),5+self.num_classes) + self.output_shape)
		output = T.set_subtensor(output[:,:,4], T.nnet.sigmoid(output[:,:,4]))
		output = T.set_subtensor(output[:,:,-self.num_classes:], softmax(output[:,:,-self.num_classes:], axis=2))
		self.output = output
		
		return
	
	def get_params(self):
		return layers.get_all_param_values(self.network['detection'])

	def set_params(self, params):
		layers.set_all_param_values(self.network['detection'], params)
		return

	'''
	Implement function for the BaseLearningObject class
	'''
	def get_weights(self):
		return [p.get_value() for p in self.get_params()]

	def get_hyperparameters(self):
		self._hyperparameters = super(YoloObjectDetector, self).get_hyperparameters()
		curr_hyperparameters = self._hyperparameters[-1]
		curr_hyperparameters.update({
			'num_classes': self.num_classes,
			'boxes': self.boxes
			})
		return self._hyperparameters

	def get_architecture(self):
		architecture = {}
		for layer in self.network:
			architecture[layer] = self.network[layer].__str__()
		return architecture

	def load_model(self, weights):
		if weights is not None:
			self.set_params(weights)
		return

	def train(self):
		# get settings from Yolo settings object
		gen_fn = self.settings.gen_fn
		train_annotations = self.settings.train_annotations
		test_annotations = self.settings.test_annotations
		train_args = self.settings.train_args
		test_args = self.settings.test_args
		print_obj = self.settings.print_obj
		update_fn = self.settings.update_fn
		update_args = self.settings.update_args
		lambda_obj = self.settings.lambda_obj
		lambda_noobj = self.settings.lambda_noobj
		lambda_noobj_coord = self.settings.lambda_noobj_coord
		rescore = self.settings.rescore
		
		if not hasattr(self, '_train_fn') or not hasattr(self, '_test_fn'):
			if not hasattr(self, 'target'):
				self.target = T.matrix('target')

			print_obj.println('Getting cost...\n')
			ti = time.time()
			cost, constants = self._get_cost(self.output, self.target, self.S, self.B, self.num_classes, rescore=rescore,
				lambda_obj=lambda_obj, lambda_noobj=lambda_noobj, lambda_noobj_coord=lambda_noobj_coord)
			cost, _ = self._get_cost(self.output_test, self.target, self.S, self.B, self.num_classes, rescore=rescore,
				lambda_obj=lambda_obj, lambda_noobj=lambda_noobj, lambda_noobj_coord=lambda_noobj_coord)
			
			print_obj.println("Creating cost variable took %.4f seconds\n" % (time.time() - ti,))

			grads = T.grad(cost, self.params, consider_constant=constants)
			updates = update_fn(grads, self.params, **update_args)
			
			print_obj.println('Compiling...\n')
			ti = time.time()
			self._train_fn = theano.function([self.input, self.target], cost, updates=updates)
			self._test_fn = theano.function([self.input, self.target], cost_test)
			
			print_obj.println('Compiling functions took %.4f seconds\n' % (time.time() - ti,))

		print_obj.println('Beginning training...\n')

		train_loss_batch = []
		test_loss_batch = []

		for Xbatch, ybatch in gen_fn(train_annotations, **train_args):
			err = self._train_fn(Xbatch, ybatch)
			train_loss_batch.append(err)
			print_obj.println('Batch error: %.4f\n' % err)

		for Xbatch, ybatch in gen_fn(test_annotations, **test_args):
			test_loss_batch.append(self._test_fn(Xbatch, ybatch))

		train_loss = np.mean(train_loss_batch)
		test_loss = np.mean(test_loss_batch)
		
		print_obj.println('\n------\nTrain Loss: %.4f, Test Loss: %.4f\n' % (train_loss, test_loss))

		return train_loss, test_loss

	def detect(self, im, thresh=0.75):
		return []

	def _get_cost(
			self,
			output,
			truth,
			lambda_obj=5.,
			lambda_noobj=1.,
			lambda_noobj_coord = 1.,
			rescore=True
		):
		cost = 0.
		
		# penalize everything, this will be undone if box matches ground truth
		cost += lambda_noobj_coord * T.mean(output[:,:,:4]**2)
		cost += lambda_noobj * T.mean(output[:,:,4]**2)
		
		# get index for each truth
		row_idx = T.cast(T.floor((truth[:,:,0] + 0.5 * truth[:,:,2]) * self.output_shape[1]), 'int32')
		col_idx = T.cast(T.floor((truth[:,:,1] + 0.5 * truth[:,:,3]) * self.output_shape[0]), 'int32')
				
		# image index
		img_idx = T.repeat(T.arange(truth.shape[0]).dimshuffle(0,'x'), truth.shape[1], axis=1)
		
		# index for each object in an image
		obj_idx = T.repeat(T.arange(truth.shape[1]), truth.shape[0], axis=0)
		
		# reshape to flat
		row_idx = row_idx.reshape((-1,))
		col_idx = col_idx.reshape((-1,))
		img_idx = img_idx.reshape((-1,))
		obj_idx = obj_idx.reshape((-1,))
		
		# use only valid indices (i.e. greater or equal to zero)
		valid_idx = T.bitwise_and(row_idx >= 0, col_idx >= 0).reshape((-1,))
		row_idx = row_idx[valid_idx.nonzero()]
		col_idx = col_idx[valid_idx.nonzero()]
		img_idx = img_idx[valid_idx.nonzero()]
		obj_idx = obj_idx[valid_idx.nonzero()]
				
		# reshape output and truth
		output = output.dimshuffle(0,'x',1,2,3,4)
		truth = truth.dimshuffle(0,1,'x',2,'x','x')
		
		output = T.repeat(output, truth.shape[1], axis=1)
		truth = T.repeat(truth, self.boxes.__len__(), axis=2)
		truth = T.repeat(T.repeat(truth, self.output_shape[0], axis=4), self.output_shape[1], axis=5)
		
		# reformat ground truth labels so that they are relative to offsets
		# and that the width/height are log scale relative to the box height.
		
		# add offset to the x,y coordinates
		x_diff, y_diff = 1./self.output_shape[0], 1./self.output_shape[1]
		y, x = meshgrid(T.arange(0 + x_diff/2,1,x_diff), T.arange(0 + y_diff/2,1,y_diff))
		x, y = x.dimshuffle('x','x',0,1), y.dimshuffle('x','x',0,1)
		
		# scaling from each anchor box
		x_scale = theano.shared(np.asarray([b[0] for b in self.boxes]), name='x_scale', borrow=True).dimshuffle('x',0,'x','x')
		y_scale = theano.shared(np.asarray([b[1] for b in self.boxes]), name='y_scale', borrow=True).dimshuffle('x',0,'x','x')

		# reformat truth
		truth = T.set_subtensor(truth[:,:,:,0,:,:], truth[:,:,:,0,:,:] - x)
		truth = T.set_subtensor(truth[:,:,:,1,:,:], truth[:,:,:,1,:,:] - y)
		truth = T.set_subtensor(truth[:,:,:,2,:,:], T.log(truth[:,:,:,2,:,:] / x_scale))
		truth = T.set_subtensor(truth[:,:,:,2,:,:], T.log(truth[:,:,:,3,:,:] / y_scale))
	
		
		# determine iou of chosen boxes
		xi = T.maximum(output[img_idx, obj_idx, :, 0, row_idx, col_idx], truth[img_idx, obj_idx, :, 0, row_idx, col_idx])
		yi = T.maximum(output[img_idx, obj_idx, :, 1, row_idx, col_idx], truth[img_idx, obj_idx, :, 1, row_idx, col_idx])
		xf = T.minimum(
			output[img_idx, obj_idx, :, 0, row_idx, col_idx] + output[img_idx, obj_idx, :, 2, row_idx, col_idx],
			truth[img_idx, obj_idx, :, 0, row_idx, col_idx] + truth[img_idx, obj_idx, :, 2, row_idx, col_idx]
		)
		yf = T.minimum(
			output[img_idx, obj_idx, :, 1, row_idx, col_idx] + output[img_idx, obj_idx, :, 3, row_idx, col_idx],
			truth[img_idx, obj_idx, :, 1, row_idx, col_idx] + truth[img_idx, obj_idx, :, 3, row_idx, col_idx]
		)
		w, h = T.maximum(xf - xi, 0.), T.maximum(yf - yi, 0.)
		
		isec = w * h
		iou = isec / (output[img_idx, obj_idx, :, 2, row_idx, col_idx] * output[img_idx, obj_idx, :, 3, row_idx, col_idx] + \
					truth[img_idx, obj_idx, :, 2, row_idx, col_idx] * truth[img_idx, obj_idx, :, 3, row_idx, col_idx] - isec)
					 
		# get index for matched boxes
		match_idx = T.argmax(iou, axis=1)
		
		# add to cost boxes which have been matched
		
		# correct for matched boxes
		cost -= lambda_noobj_coord * T.mean(output[img_idx, obj_idx, :, :4, row_idx, col_idx][:,match_idx]**2)
		cost -= lambda_noobj * T.mean(output[img_idx, obj_idx, :, 4, row_idx, col_idx][:,match_idx]**2)
		
		# coordinate errors
		cost += lambda_obj * T.mean(
			(output[img_idx, obj_idx, :, 0, row_idx, col_idx][:,match_idx] - truth[img_idx, obj_idx, :, 0, row_idx, col_idx][:,match_idx])**2
		)
		cost += lambda_obj * T.mean(
			(output[img_idx, obj_idx, :, 1, row_idx, col_idx][:,match_idx] - truth[img_idx, obj_idx, :, 1, row_idx, col_idx][:,match_idx])**2
		)
		cost += lambda_obj * T.mean(
			(output[img_idx, obj_idx, :, 2, row_idx, col_idx][:,match_idx] - truth[img_idx, obj_idx, :, 2, row_idx, col_idx][:,match_idx])**2
		)
		cost += lambda_obj * T.mean(
			(output[img_idx, obj_idx, :, 3, row_idx, col_idx][:,match_idx] - truth[img_idx, obj_idx, :, 3, row_idx, col_idx][:,match_idx])**2
		)
		
		# objectness error
		if rescore:
			cost += lambda_obj * T.mean(
				(output[img_idx, obj_idx, :, 3, row_idx, col_idx][:,match_idx] - iou[:,match_idx])**2
			)
		else:
			cost += lambda_obj * T.mean(
				(output[img_idx, obj_idx, :, 3, row_idx, col_idx][:,match_idx])**2
			)
		
		# class error
		cost += lambda_obj * T.mean(
			(
				-truth[img_idx, obj_idx, :, -self.num_classes:, row_idx, col_idx][:,match_idx] * \
				T.log(output[img_idx, obj_idx, :, -self.num_classes:, row_idx, col_idx][:,match_idx])
			)
		)
				
		return cost