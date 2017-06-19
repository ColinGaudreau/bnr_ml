import theano
from theano import tensor as T
import numpy as np

from bnr_ml.nnet.updates import momentum as momentum_update
from bnr_ml.nnet.layers import AbstractNNetLayer
from bnr_ml.utils.helpers import meshgrid, bitwise_not, StreamPrinter, format_image
from bnr_ml.utils.nonlinearities import softmax, smooth_l1, smooth_abs, safe_sqrt
from bnr_ml.utils.theano_extensions import argmin_unique
from bnr_ml.objectdetect import utils
from bnr_ml.objectdetect.nms import nms
from bnr_ml.logger.learning_objects import BaseLearningObject, BaseLearningSettings
from yolo2_cost import yolo2_cost


from collections import OrderedDict
from tqdm import tqdm
import time
from PIL import Image, ImageDraw

from lasagne import layers
from lasagne.updates import rmsprop, sgd, adam
from lasagne.updates import momentum as momentum_update

from itertools import tee

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
			lambda_noobj=0.5,
			lambda_anchor=.1,
			thresh=0.8,
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
		self.lambda_anchor = lambda_anchor
		self.thresh = thresh
		self.rescore = rescore
		self.hyperparameters = {}

	def serialize(self):
		serialization = {}
		serialization['update_fn'] = self.update_fn.__str__()
		serialization['update_args'] = self.update_args
		serialization['lambda_obj'] = self.lambda_obj
		serialization['lambda_noobj'] = self.lambda_noobj
		serialization['lambda_anchor'] = self.lambda_anchor
		serialization['thresh'] = self.thresh
		serialization['rescore'] = self.rescore  
		serialization.update(self.hyperparameters)
		return serialization

class Yolo2ObjectDetector(BaseLearningObject):
	def __init__(
		self,
		network,
		num_classes,
		boxes=[(5.,.5), (.25,.5), (.5,.25), (.3,.3)],
		use_custom_cost=False
	):
		assert('detection' in network and 'input' in network)
		super(Yolo2ObjectDetector, self).__init__()	
		self.network = network
		self.num_classes = num_classes
		self.boxes = boxes
		self.use_custom_cost = use_custom_cost
		
		self.input = network['input'].input_var
		self.output_shape = layers.get_output_shape(network['detection'])[-2:]
		self.input_shape = network['input'].shape[-2:]
#		 self._hyperparameters = {'ratios': ratios, 'smin': smin, 'smax': smax}
		
		
		# set objectness predictor nonlinearity to sigmoid and
		# class predictor to softmax
		self.output = self._format_output(layers.get_output(network['detection'], deterministic=False))
		self.output_test = self._format_output(layers.get_output(network['detection'], deterministic=True))
		
		return

	def _format_output(self, output):
		if self.use_custom_cost:
			obj_idx = T.arange(4, output.shape[1], 5+self.num_classes)
			output = T.set_subtensor(output[:,obj_idx], T.nnet.sigmoid(output[:,obj_idx]))
			
			for i in range(len(self.boxes)):
				cls_idx = T.arange(self.num_classes) + 5 + i * (5 + self.num_classes)
				output = T.set_subtensor(output[:,cls_idx], softmax(output[:,cls_idx], axis=1))
			
			return output
		else:
			# for old cost, gonna not do that cus I am meaaaaaan
			output = T.reshape(output, (-1,self.boxes.__len__(),5+self.num_classes) + self.output_shape)
			output = T.set_subtensor(output[:,:,4], T.nnet.sigmoid(output[:,:,4]))
			output = T.set_subtensor(output[:,:,-self.num_classes:], softmax(output[:,:,-self.num_classes:], axis=2))
			return output
	
	def get_params(self):
		return layers.get_all_params(self.network['detection'])

	def set_params(self, params):
		layers.set_all_param_values(self.network['detection'], params)
		return

	'''
	Implement function for the BaseLearningObject class
	'''
	def get_weights(self):
		return [p.get_value() for p in self.get_params()]

	def get_hyperparameters(self):
		self._hyperparameters = super(Yolo2ObjectDetector, self).get_hyperparameters()
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

	def train(self, recompile=False):
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
		lambda_anchor = self.settings.lambda_anchor
		thresh = self.settings.thresh
		rescore = self.settings.rescore
		use_custom_cost = self.use_custom_cost
		
		if not hasattr(self, '_train_fn') or not hasattr(self, '_test_fn') or recompile:
			if not hasattr(self, 'target'):
				self.target = T.tensor3('target')

			print_obj.println('Getting cost...\n')
			ti = time.time()

			if use_custom_cost:
				constants = []
				cost, anchors =  yolo2_cost(self.output, self.target, self.num_classes, len(self.boxes), lambda_obj, lambda_noobj, self.boxes, return_anchors=True)
				cost_test, _ =  yolo2_cost(self.output_test, self.target, self.num_classes, len(self.boxes), lambda_obj, lambda_noobj, self.boxes, return_anchors=True)
			else:
				cost, constants, extras = self._get_cost(self.output, self.target, rescore=rescore)
				cost_test, _, _ = self._get_cost(self.output_test, self.target, rescore=rescore)
			
			print_obj.println("Creating cost variable took %.4f seconds\n" % (time.time() - ti,))
			
			params = self.get_params()
			grads = T.grad(cost, params, consider_constant=constants)
			updates = update_fn(grads, params, **update_args)
			
			print_obj.println('Compiling...\n')
			ti = time.time()
			if use_custom_cost:
				self._train_fn = theano.function([self.input, self.target], [cost, anchors], updates=updates)
				self._test_fn = theano.function([self.input, self.target], cost_test)
			else:
				output_vars = [cost]
				output_vars.extend(extras)
				self._train_fn = theano.function([self.input, self.target, self._lambda_obj, self._lambda_noobj, self._lambda_anchor], output_vars, updates=updates)
				self._test_fn = theano.function([self.input, self.target, self._lambda_obj, self._lambda_noobj, self._lambda_anchor], cost_test)
			
			print_obj.println('Compiling functions took %.4f seconds\n' % (time.time() - ti,))

		print_obj.println('Beginning training...\n')

		if use_custom_cost:
			train_loss_batch = []
			test_loss_batch = []
			extras = {'anchors': []}

			for Xbatch, ybatch in gen_fn(train_annotations, **train_args):
				ret_args = self._train_fn(Xbatch, ybatch)
				err = ret_args[0]
				extras['anchors'].extend(np.asarray(ret_args[1]).tolist())
				train_loss_batch.append(err)
				print_obj.println('Batch error: %.4f' % err)

			for Xbatch, ybatch in gen_fn(test_annotations, **test_args):
				test_loss_batch.append(self._test_fn(Xbatch, ybatch))
			
			train_loss = np.mean(train_loss_batch)
			test_loss = np.mean(test_loss_batch)
			
			print_obj.println('\n------\nTrain Loss: %.4f, Test Loss: %.4f\n' % (train_loss, test_loss))
		else:	
			train_loss_batch = []
			test_loss_batch = []
			extras = {'rows': [], 'cols': [], 'anchors': []}
			cost_breakdown = []

			for Xbatch, ybatch in gen_fn(train_annotations, **train_args):
				ret_args = self._train_fn(Xbatch, ybatch, lambda_obj, lambda_noobj, lambda_anchor)
				err = ret_args[0]
				train_loss_batch.append(err)
				extras['rows'].extend(ret_args[1].tolist())
				extras['cols'].extend(ret_args[2].tolist())
				extras['anchors'].extend(ret_args[3].tolist())
				cost_breakdown.append(ret_args[-5:])
				print_obj.println('Batch error: %.4f' % err)

			# log the breakdown of the cost function
			cost_breakdown = [float(c) for c in np.mean(np.asarray(cost_breakdown), axis=0)]	
			extras['cost_noobject'] = cost_breakdown[0]
			extras['cost_anchor'] = cost_breakdown[1]
			extras['cost_coord'] = cost_breakdown[2]
			extras['cost_class'] = cost_breakdown[3]
			extras['cost_obj'] = cost_breakdown[4]

			for Xbatch, ybatch in gen_fn(test_annotations, **test_args):
				test_loss_batch.append(self._test_fn(Xbatch, ybatch, lambda_obj, lambda_noobj, lambda_anchor))

			train_loss = np.mean(train_loss_batch)
			test_loss = np.mean(test_loss_batch)
			
			print_obj.println('\n------\nTrain Loss: %.4f, Test Loss: %.4f\n' % (train_loss, test_loss))

		return float(train_loss), float(test_loss), extras



	def detect(self, im, thresh=0.75, overlap=0.5, num_to_label=None):
		im = format_image(im, dtype=theano.config.floatX)

		old_size = im.shape[:2]
		im = cv2.resize(im, self.input_shape[::-1], interpolation=cv2.INTER_LINEAR).swapaxes(2,1).swapaxes(1,0).reshape((1,3) + self.input_shape)

		if not hasattr(self, '_detect_fn'):
			'''
			Make theano do all the heavy lifting for detection, this should speed up the process marginally.
			'''

			output = self.output_test

			if self.use_custom_cost:
				new_output = None
				for i in range(len(self.boxes)):
					cls_idx = T.arange(i * (5 + self.num_classes), (i+1) * (5 + self.num_classes))
					if new_output is None:
						new_output = output[:,cls_idx,:,:].dimshuffle(0,'x',1,2,3)
					else:
						new_output = T.concatenate((new_output, output[:,cls_idx,:,:].dimshuffle(0,'x',1,2,3)), axis=1)
				output = new_output

			thresh_var = T.scalar(name='thresh')
			conf = output[:,:,4] * T.max(output[:,:,-self.num_classes:], axis=2)

			# define offsets to predictions
			w_cell, h_cell =  1. / self.output_shape[1], 1. / self.output_shape[0]
			x, y = T.arange(w_cell / 2, 1., w_cell), T.arange(h_cell / 2, 1., h_cell)
			y, x = meshgrid(x, y)

			x, y = x.dimshuffle('x','x',0,1), y.dimshuffle('x','x',0,1)
			
			# define scale
			w_acr = theano.shared(np.asarray([b[0] for b in self.boxes]), name='w_acr', borrow=True).dimshuffle('x',0,'x','x')
			h_acr = theano.shared(np.asarray([b[1] for b in self.boxes]), name='h_acr', borrow=True).dimshuffle('x',0,'x','x')

			# rescale output
			output = T.set_subtensor(output[:,:,2], w_acr * T.exp(output[:,:,2]))
			output = T.set_subtensor(output[:,:,3], h_acr * T.exp(output[:,:,3]))
			output = T.set_subtensor(output[:,:,0], output[:,:,0] + x - output[:,:,2] / 2)
			output = T.set_subtensor(output[:,:,1], output[:,:,1] + y - output[:,:,3] / 2)

			# define confidence in prediction
			conf = output[:,:,4] * T.max(output[:,:,-self.num_classes:], axis=2)
			cls = T.argmax(output[:,:,-self.num_classes:], axis=2)

			# filter out all below thresh
			above_thresh_idx = conf > thresh_var			
			pred = T.concatenate(
				(
					output[:,:,0][above_thresh_idx.nonzero()].dimshuffle(0,'x'),
					output[:,:,1][above_thresh_idx.nonzero()].dimshuffle(0,'x'),
					output[:,:,2][above_thresh_idx.nonzero()].dimshuffle(0,'x'),
					output[:,:,3][above_thresh_idx.nonzero()].dimshuffle(0,'x'),
					conf[above_thresh_idx.nonzero()].dimshuffle(0,'x'),
					cls[above_thresh_idx.nonzero()].dimshuffle(0,'x')
				),
				axis=1
			)
			
			self._detect_fn = theano.function([self.input, thresh_var], pred)

		output = self._detect_fn(im, thresh)

		boxes = []
		for i in range(output.shape[0]):
			coord, conf, cls = output[i,:4], output[i,4], output[i,5]
			coord[2:] += coord[:2]
			if num_to_label is not None:
				cls =num_to_label[cls]
			box = utils.BoundingBox(*coord.tolist(), confidence=conf, cls=cls)
			boxes.append(box)

		return boxes

	def _get_cost2(
			self,
			output,
			truth,
			rescore=True
		):

		if not hasattr(self, '_lambda_obj'):
			lambda_obj, lambda_noobj, thresh = T.scalar('lambda_obj'), T.scalar('lambda_noobj'), T.scalar('thresh')
			self._lambda_obj, self._lambda_noobj, self._thresh = lambda_obj, lambda_noobj, thresh
		else:
			lambda_obj, lambda_noobj, thresh = self._lambda_obj, self._lambda_noobj, self._thresh
		
		cost = 0.
		# create grid for cells
		w_cell, h_cell =  1. / self.output_shape[1], 1. / self.output_shape[0]
		x, y = T.arange(w_cell / 2, 1., w_cell), T.arange(h_cell / 2, 1., h_cell)
		y, x = meshgrid(x, y)
		
		# reshape truth to match with cell
		truth_cell = truth.dimshuffle(0, 1, 2, 'x','x')
		x, y = x.dimshuffle('x','x',0,1), y.dimshuffle('x','x',0,1)
		
		# calculate overlap between cell and ground truth boxes
		xi, yi = T.maximum(truth_cell[:,:,0], x - w_cell/2), T.maximum(truth_cell[:,:,1], y - h_cell/2)
		xf = T.minimum(truth_cell[:,:,[0,2]].sum(axis=2), x + w_cell/2)
		yf = T.minimum(truth_cell[:,:,[1,3]].sum(axis=2), y + h_cell/2)
		w, h = T.maximum(xf - xi, 0), T.maximum(yf - yi, 0)
		
		# overlap between cell and ground truth box
		overlap = (w * h) / (w_cell * h_cell)
		
		# repeat truth boxes
		truth_boxes = truth.dimshuffle(0, 1, 'x', 2, 'x', 'x')
		
		# create grid for anchor boxes
		anchors = T.concatenate((x.dimshuffle(0,1,'x','x',2,3) - w_cell/2, y.dimshuffle(0,1,'x','x',2,3) - h_cell/2), axis=3)
		anchors = T.concatenate((anchors, T.ones_like(anchors)), axis=3)
		anchors = T.repeat(anchors, self.boxes.__len__(), axis=2)
		
		w_acr = theano.shared(np.asarray([b[0] for b in self.boxes]), name='w_acr', borrow=True).dimshuffle('x','x',0,'x','x')
		h_acr = theano.shared(np.asarray([b[1] for b in self.boxes]), name='h_acr', borrow=True).dimshuffle('x','x',0,'x','x')
		
		anchors = T.set_subtensor(anchors[:,:,:,2], anchors[:,:,:,2] * w_acr)
		anchors = T.set_subtensor(anchors[:,:,:,3], anchors[:,:,:,3] * h_acr)
		
		# find iou between anchors and ground truths
		xi, yi = T.maximum(truth_boxes[:,:,:,0], anchors[:,:,:,0]), T.maximum(truth_boxes[:,:,:,1], anchors[:,:,:,1])
		xf = T.minimum(truth_boxes[:,:,:,[0,2]].sum(axis=3), anchors[:,:,:,[0,2]].sum(axis=3))
		yf = T.minimum(truth_boxes[:,:,:,[1,3]].sum(axis=3), anchors[:,:,:,[1,3]].sum(axis=3))
		w, h = T.maximum(xf - xi, 0), T.maximum(yf - yi, 0)
		
		isec = w * h
		iou = isec / (T.prod(truth_boxes[:,:,:,[2,3]], axis=3) + T.prod(anchors[:,:,:,[2,3]], axis=3) - isec)
		
		overlap = overlap.dimshuffle(0,1,'x',2,3)
		
		best_iou_obj_idx = T.argmax(iou, axis=1).dimshuffle(0,'x',1,2,3)
		best_iou_box_idx = T.argmax(iou, axis=2).dimshuffle(0,1,'x',2,3)
		
		_,obj_idx,box_idx,_,_ = meshgrid(
			T.arange(truth.shape[0]),
			T.arange(truth.shape[1]),
			T.arange(self.boxes.__len__()),
			T.arange(self.output_shape[0]),
			T.arange(self.output_shape[1])
		)
		
		# define logical matrix assigning object to correct anchor box and cell.
		best_iou_idx = T.bitwise_and(
			T.bitwise_and(
				T.eq(best_iou_box_idx, box_idx),
				T.eq(best_iou_obj_idx, obj_idx)
			),
			overlap >= thresh
		)
		
		constants = []
		if rescore: 
			# scale predictions correctly
			pred = output.dimshuffle(0,'x',1,2,3,4)
			pred = T.set_subtensor(pred[:,:,:,0], pred[:,:,:,0] + x.dimshuffle(0,1,'x',2,3))
			pred = T.set_subtensor(pred[:,:,:,1], pred[:,:,:,1] + y.dimshuffle(0,1,'x',2,3))
			pred = T.set_subtensor(pred[:,:,:,2], w_acr * T.exp(pred[:,:,:,2]))
			pred = T.set_subtensor(pred[:,:,:,3], h_acr * T.exp(pred[:,:,:,3]))
			
			xi, yi = T.maximum(pred[:,:,:,0], truth_boxes[:,:,:,0]), T.maximum(pred[:,:,:,1], truth_boxes[:,:,:,1])
			xf = T.minimum(pred[:,:,:,[0,2]].sum(axis=3), truth_boxes[:,:,:,[0,2]].sum(axis=3))
			yf = T.minimum(pred[:,:,:,[1,3]].sum(axis=3), truth_boxes[:,:,:,[1,3]].sum(axis=3))
			w, h = T.maximum(xf - xi, 0.), T.maximum(yf - yi, 0.)
			
			isec = w * h
			iou = isec / (pred[:,:,:,[2,3]].prod(axis=3) + truth_boxes[:,:,:,[2,3]].prod(axis=3) - isec)

			# make sure iou is considered constant when taking gradient
			constants.append(iou)
	
		# format ground truths correclty
		truth_boxes = truth_boxes = T.repeat(
			T.repeat(
				T.repeat(truth_boxes, self.boxes.__len__(), axis=2),
				self.output_shape[0], axis=4
			),
			self.output_shape[1], axis=5
		)
		
		truth_boxes = T.set_subtensor(truth_boxes[:,:,:,0], truth_boxes[:,:,:,0] - anchors[:,:,:,0])
		truth_boxes = T.set_subtensor(truth_boxes[:,:,:,1], truth_boxes[:,:,:,1] - anchors[:,:,:,1])
		truth_boxes = T.set_subtensor(truth_boxes[:,:,:,2], T.log(truth_boxes[:,:,:,2] / anchors[:,:,:,2]))
		truth_boxes = T.set_subtensor(truth_boxes[:,:,:,3], T.log(truth_boxes[:,:,:,3] / anchors[:,:,:,3]))
		
		# add dimension for objects per image
		pred = T.repeat(output.dimshuffle(0,'x',1,2,3,4), truth.shape[1], axis=1)
				
		# penalize coordinates
		cost += lambda_obj * T.mean(((pred[:,:,:,:4] - truth_boxes[:,:,:,:4])**2).sum(axis=3)[best_iou_idx.nonzero()])
				
		# penalize class scores
		cost += lambda_obj * T.mean((-truth_boxes[:,:,:,-self.num_classes:] * T.log(pred[:,:,:,-self.num_classes:])).sum(axis=3)[best_iou_idx.nonzero()])
		
		# penalize objectness score
		if rescore:
			cost += lambda_obj * T.mean(((pred[:,:,:,4] - iou)**2)[best_iou_idx.nonzero()])
		else:
			cost += lambda_obj * T.mean(((pred[:,:,:,4] - 1.)**2)[best_iou_idx.nonzero()])
		
		# flip all matched and penalize all un-matched objectness scores
		not_matched_idx = best_iou_idx.sum(axis=1) > 0
		not_matched_idx = bitwise_not(not_matched_idx)

		# penalize objectness score for non-matched boxes
		cost += lambda_noobj * T.mean((pred[:,0,:,4]**2)[not_matched_idx.nonzero()])
		
		return cost, constants

	def _get_cost3(
			self,
			output,
			truth,
			rescore=True
		):

		if not hasattr(self, '_lambda_obj'):
			lambda_obj, lambda_noobj = T.scalar('lambda_obj'), T.scalar('lambda_noobj')
			self._lambda_obj, self._lambda_noobj = lambda_obj, lambda_noobj
		else:
			lambda_obj, lambda_noobj, thresh = self._lambda_obj, self._lambda_noobj, self._thresh

		cost = 0.
		
		# penalize everything, this will be undone if box matches ground truth
		#cost += lambda_noobj_coord * T.mean(output[:,:,:4]**2)
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

		# change predicted output to proper scale
		pred = T.set_subtensor(output[:,:,:,0], output[:,:,:,0] + x)
		pred = T.set_subtensor(pred[:,:,:,1], pred[:,:,:,1] + y)
		pred = T.set_subtensor(pred[:,:,:,2], x_scale * T.exp(pred[:,:,:,2]))
		pred = T.set_subtensor(pred[:,:,:,3], y_scale * T.exp(pred[:,:,:,3]))
		
		# determine iou of chosen boxes
		xi = T.maximum(pred[img_idx, obj_idx, :, 0, row_idx, col_idx], truth[img_idx, obj_idx, :, 0, row_idx, col_idx])
		yi = T.maximum(pred[img_idx, obj_idx, :, 1, row_idx, col_idx], truth[img_idx, obj_idx, :, 1, row_idx, col_idx])
		xf = T.minimum(
			pred[img_idx, obj_idx, :, 0, row_idx, col_idx] + pred[img_idx, obj_idx, :, 2, row_idx, col_idx],
			truth[img_idx, obj_idx, :, 0, row_idx, col_idx] + truth[img_idx, obj_idx, :, 2, row_idx, col_idx]
		)
		yf = T.minimum(
			pred[img_idx, obj_idx, :, 1, row_idx, col_idx] + pred[img_idx, obj_idx, :, 3, row_idx, col_idx],
			truth[img_idx, obj_idx, :, 1, row_idx, col_idx] + truth[img_idx, obj_idx, :, 3, row_idx, col_idx]
		)
		w, h = T.maximum(xf - xi, 0.), T.maximum(yf - yi, 0.)
		
		isec = w * h
		iou = isec / (pred[img_idx, obj_idx, :, 2, row_idx, col_idx] * pred[img_idx, obj_idx, :, 3, row_idx, col_idx] + \
					truth[img_idx, obj_idx, :, 2, row_idx, col_idx] * truth[img_idx, obj_idx, :, 3, row_idx, col_idx] - isec)
					 
		# get index for matched boxes
		match_idx = T.argmax(iou, axis=1)
		
		# change truth to proper scale for error
		truth = T.set_subtensor(truth[:,:,:,0,:,:], truth[:,:,:,0,:,:] - x)
		truth = T.set_subtensor(truth[:,:,:,1,:,:], truth[:,:,:,1,:,:] - y)
		truth = T.set_subtensor(truth[:,:,:,2,:,:], T.log(truth[:,:,:,2,:,:] / x_scale))
		truth = T.set_subtensor(truth[:,:,:,3,:,:], T.log(truth[:,:,:,3,:,:] / y_scale))
		
		# add to cost boxes which have been matched
		
		# correct for matched boxes
		#cost -= lambda_noobj_coord * T.mean(output[img_idx, obj_idx, :, :4, row_idx, col_idx][:,match_idx]**2)
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
				(output[img_idx, obj_idx, :, 4, row_idx, col_idx][:,match_idx] - iou[:,match_idx])**2
			)
		else:
			cost += lambda_obj * T.mean(
				(output[img_idx, obj_idx, :, 4, row_idx, col_idx][:,match_idx] - 1)**2
			)
		
		# class error
		cost += lambda_obj * T.mean(
			(
				-truth[img_idx, obj_idx, :, -self.num_classes:, row_idx, col_idx][:,match_idx] * \
				T.log(output[img_idx, obj_idx, :, -self.num_classes:, row_idx, col_idx][:,match_idx])
			)
		)
				
		return cost, [iou]

	def _get_cost(
			self,
			output,
			truth,
			rescore=True
		):
		if not hasattr(self, '_lambda_obj'):
			lambda_obj, lambda_noobj, lambda_anchor = T.scalar('lambda_obj'), T.scalar('lambda_noobj'), T.scalar('lambda_anchor')
			self._lambda_obj, self._lambda_noobj, self._lambda_anchor = lambda_obj, lambda_noobj, lambda_anchor
		else:
			lambda_obj, lambda_noobj, lambda_anchor = self._lambda_obj, self._lambda_noobj, self._lambda_anchor
			
		# lambda_obj, lambda_noobj, lambda_anchor = 1., 5., 0.1

		w_cell, h_cell = 1./self.output_shape[1], 1./self.output_shape[0]
		x, y = T.arange(w_cell/2, 1., w_cell), T.arange(h_cell/2, 1., h_cell)
		y, x = meshgrid(x, y)
		x, y = x.dimshuffle('x','x','x',0,1), y.dimshuffle('x','x','x',0,1)

		# create anchors for later
		w_acr = theano.shared(np.asarray([b[0] for b in self.boxes]), name='w_acr').dimshuffle('x',0,'x','x','x') * T.ones_like(x)
		h_acr = theano.shared(np.asarray([b[1] for b in self.boxes]), name='h_acr').dimshuffle('x',0,'x','x','x') * T.ones_like(y)
		anchors = T.concatenate((x * T.ones_like(w_acr), y * T.ones_like(h_acr), w_acr, h_acr), axis=2)
		anchors = T.repeat(anchors, truth.shape[0], axis=0)

		cell_coord = T.concatenate((x,y), axis=2)
		gt_coord = (truth[:,:,:2] + truth[:,:,2:4]/2).dimshuffle(0,1,2,'x','x')
		
		gt_dist = T.sum((gt_coord - cell_coord)**2, axis=2).reshape((truth.shape[0],truth.shape[1],-1))
		
		cell_idx = argmin_unique(gt_dist, 1, 2).reshape((-1,)) # assign unique cell to each obj per example
		row_idx = T.cast(cell_idx // self.output_shape[1], 'int64')
		col_idx = cell_idx - row_idx * self.output_shape[1]
		num_idx = T.repeat(T.arange(truth.shape[0]).reshape((-1,1)), truth.shape[1], axis=1).reshape((-1,))
		obj_idx = T.repeat(T.arange(truth.shape[1]).reshape((1,-1)), truth.shape[0], axis=0).reshape((-1,))
		
		valid_example = gt_dist[num_idx, obj_idx, cell_idx] < 1 # if example further than 1 away from cell it's a garbage example
		
		num_idx, obj_idx = num_idx[valid_example.nonzero()], obj_idx[valid_example.nonzero()]
		row_idx, col_idx = row_idx[valid_example.nonzero()], col_idx[valid_example.nonzero()]
		
		truth_flat = truth[num_idx, obj_idx, :].dimshuffle(0,'x',1)
		
		pred_matched = output[num_idx,:,:,row_idx, col_idx]
		x, y = x[:,0,0,row_idx, col_idx].dimshuffle(1,0), y[:,0,0,row_idx, col_idx].dimshuffle(1,0)
		w_acr = theano.shared(np.asarray([b[0] for b in self.boxes]), name='w_acr').dimshuffle('x',0)
		h_acr = theano.shared(np.asarray([b[1] for b in self.boxes]), name='h_acr').dimshuffle('x',0)
	
		# reformat prediction
		pred_shift = pred_matched
		pred_shift = T.set_subtensor(pred_shift[:,:,2], w_acr * T.exp(pred_shift[:,:,2]))
		pred_shift = T.set_subtensor(pred_shift[:,:,3], h_acr * T.exp(pred_shift[:,:,3]))
		pred_shift = T.set_subtensor(pred_shift[:,:,0], pred_shift[:,:,0] + T.repeat(x, pred_shift.shape[1], axis=1) - pred_shift[:,:,2]/2)
		pred_shift = T.set_subtensor(pred_shift[:,:,1], pred_shift[:,:,1] + T.repeat(y, pred_shift.shape[1], axis=1) - pred_shift[:,:,3]/2)
		
		# calculate iou
		xi = T.maximum(pred_shift[:,:,0], truth_flat[:,:,0])
		yi = T.maximum(pred_shift[:,:,1], truth_flat[:,:,1])
		xf = T.minimum(pred_shift[:,:,[0,2]].sum(axis=2), truth_flat[:,:,[0,2]].sum(axis=2))
		yf = T.minimum(pred_shift[:,:,[1,3]].sum(axis=2), truth_flat[:,:,[1,3]].sum(axis=2))
		w, h = T.maximum(xf - xi, 0), T.maximum(yf - yi, 0)
		
		isec = w * h
		union = T.prod(pred_shift[:,:,[2,3]], axis=2) + T.prod(truth_flat[:,:,[2,3]], axis=2) - isec
		iou = isec / union

		# calculate iou for anchor
		anchors_matched = anchors[num_idx,:,:,row_idx,col_idx]
		xi = T.maximum(anchors_matched[:,:,0], truth_flat[:,:,0])
		yi = T.maximum(anchors_matched[:,:,1], truth_flat[:,:,1])
		xf = T.minimum(anchors_matched[:,:,[0,2]].sum(axis=2), truth_flat[:,:,[0,2]].sum(axis=2))
		yf = T.minimum(anchors_matched[:,:,[1,3]].sum(axis=2), truth_flat[:,:,[1,3]].sum(axis=2))
		w, h = T.maximum(xf - xi, 0), T.maximum(yf - yi, 0)
		
		isec = w * h
		union = T.prod(anchors_matched[:,:,[2,3]], axis=2) + T.prod(truth_flat[:,:,[2,3]], axis=2) - isec
		iou_acr = isec / union
		
		# get max iou
		acr_idx = T.argmax(iou_acr, axis=1)
		
		# reformat truth
		truth_formatted = truth_flat
		truth_formatted = T.repeat(truth_formatted, self.boxes.__len__(), axis=1)
		truth_formatted = T.set_subtensor(truth_formatted[:,:,0], truth_formatted[:,:,0] + truth_formatted[:,:,2]/2 - T.repeat(x, truth_formatted.shape[1], axis=1))
		truth_formatted = T.set_subtensor(truth_formatted[:,:,1], truth_formatted[:,:,1] + truth_formatted[:,:,3]/2 - T.repeat(y, truth_formatted.shape[1], axis=1))
		truth_formatted = T.set_subtensor(truth_formatted[:,:,2], T.log(truth_formatted[:,:,2] / w_acr))
		truth_formatted = T.set_subtensor(truth_formatted[:,:,3], T.log(truth_formatted[:,:,3] / h_acr))
		truth_formatted = truth_formatted[T.arange(truth_formatted.shape[0]),acr_idx,:]
		
			
		#
		# calculate cost
		#
		item_idx = T.arange(pred_matched.shape[0])
		anchors = T.set_subtensor(anchors[:,:,:2], 0.)

		cost = 0.

		cost_noobject = lambda_noobj * (T.mean(output[:,:,4]**2) - T.sum(pred_matched[item_idx, acr_idx,4]**2) / output[:,:,4].size)
		cost_anchor = lambda_anchor * (T.mean(T.sum(output[:,:,:4]**2, axis=2)) - T.sum(T.sum(pred_matched[item_idx,acr_idx,:4]**2, axis=1)) / output[:,:,0].size)
		cost_coord = lambda_obj * T.mean(T.sum((pred_matched[item_idx,acr_idx,:4] - truth_formatted[:,:4])**2, axis=1))
		cost_class = lambda_obj * T.mean(T.sum(-truth_formatted[:,-self.num_classes:] * T.log(pred_matched[item_idx, acr_idx, -self.num_classes:]), axis=1))

		if rescore:
			cost_obj = lambda_obj * T.mean((pred_matched[item_idx, acr_idx,4] - iou[item_idx, acr_idx])**2)
		else:
			cost_obj = lambda_obj * T.mean((pred_matched[item_idx, acr_idx,4] - 1)**2)
		
		cost = cost_noobject + cost_obj + cost_anchor + cost_coord + cost_class

		return cost, [iou], [row_idx, col_idx, acr_idx, cost_noobject, cost_anchor, cost_coord, cost_class, cost_obj]
