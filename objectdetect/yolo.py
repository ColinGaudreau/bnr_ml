import theano
from theano import tensor as T
import numpy as np

from bnr_ml.nnet.updates import momentum as momentum_update
from bnr_ml.nnet.layers import AbstractNNetLayer
from bnr_ml.utils.helpers import meshgrid2D, bitwise_not, StreamPrinter
from bnr_ml.utils.nonlinearities import softmax, smooth_l1, smooth_abs, safe_sqrt
from bnr_ml.objectdetect import utils

from collections import OrderedDict
from tqdm import tqdm
import time
from PIL import Image, ImageDraw

from lasagne import layers
from lasagne.updates import rmsprop, sgd, adam
from lasagne.updates import momentum as momentum_update

from itertools import tee

from ml_logger.learning_objects import BaseLearningObject, BaseLearningSettings

import pdb

class YoloSettings(BaseLearningSettings):
	def __init__(
			self,
			train_annotations,
			test_annotations,
			train_args,
			test_args=None,
			print_obj=None,
			update_fn=rmsprop,
			update_args={'learning_rate': 1e-5},
			lmbda_coord=5.,
			lmbda_noobj=0.5,
			lmbda_obj=1.,
			rescore=True,
			hyperparameters={}
		):
		super(YoloSettings, self).__init__()
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
		self.update_fn = rmsprop
		self.update_args = update_args
		self.lmbda_coord = lmbda_coord
		self.lmbda_noobj = lmbda_noobj
		self.lmbda_obj = lmbda_obj
		self.rescore = rescore
		self.hyperparameters = {}

	def serialize(self):
		serialization = {}
		serialization['update_fn'] = self.update_fn.__str__()
		serialization['lmbda_coord'] = self.lmbda_coord
		serialization['lmbda_noobj'] = self.lmbda_noobj
		serialization['lmbda_obj'] = self.lmbda_obj
		serialization['rescore'] = self.rescore	
		serialization.extend(self.hyperparameters)
		return serialization

class YoloObjectDetector(BaseLearningObject):

	def __init__(
			self,
			network,
			input_shape,
			num_classes,
			S,
			B
		):
		'''
		network:
		--------
			Dict with the entire network defined, must have a "feature_map" and "output" layer.
			You must be able to call .get_output() on these layers.
		'''
		super(YoloObjectDetector, self).__init__()
		self.network = network
		self.num_classes = num_classes
		self.S = S
		self.B = B
		self.input = network['input'].input_var
		self.input_shape = input_shape

		output = layers.get_output(network['output'], deterministic=False)
		output_test = layers.get_output(network['output'], deterministic=True)

		def get_output(output, B, S, num_classes):
			output = T.reshape(output, (-1, B * 5 + num_classes, S[0], S[1]))
			for i in range(B):
				#output = T.set_subtensor(output[:,5*i:5*i+2,:,:], 2 * T.nnet.sigmoid(output[:,5*i:5*i+2,:,:]) - 1)
				output = T.set_subtensor(output[:,5*i + 2:5*i + 4,:,:], T.abs_(output[:,5*i + 2:5*i + 4,:,:]))
				#output = T.set_subtensor(output[:,5*i + 4,:,:], T.nnet.sigmoid(output[:,5*i + 4,:,:]))
				pass
			output = T.set_subtensor(output[:,-self.num_classes:,:,:], softmax(output[:,-self.num_classes:,:,:], axis=1)) # use safe softmax
			return output
		self.output = get_output(output, B, S, num_classes)
		self.output_test = get_output(output_test, B, S, num_classes)

		self.params = layers.get_all_params(network['output'])

	def _get_cost(self, output, truth, S, B, C, rescore=False, lmbda_coord=5., lmbda_noobj=0.5, lmbda_obj=1., iou_thresh=1e-5):
		'''
		Calculates cost for multiple objects in a scene without for loops or scan (so reduces the amount of variable
		created in the theano computation graph).  A cell is associated with a certain object if the iou of that cell
		and the object is higher than any other ground truth object. and the rest of the objectness scores are pushed
		towards zero.

		Returns the cost and list of variable that I don't want to backpropagate through.
		'''
		
		# calculate height/width of individual cell
		block_height, block_width = 1. / S[0], 1./ S[1]

		# get the offset of each cell
		offset_x, offset_y = meshgrid2D(T.arange(0,1,block_width), T.arange(0,1,block_height))

		# get indices for x,y,w,h,object-ness for easy access
		x_idx, y_idx = T.arange(0,5*B,5), T.arange(1,5*B, 5)
		w_idx, h_idx = T.arange(2,5*B,5), T.arange(3,5*B,5)
		conf_idx = T.arange(4,5*B,5)

		# Get position predictions with offsets.
		pred_x = (output[:,x_idx] + offset_x.dimshuffle('x','x',0,1)).dimshuffle(0,'x',1,2,3)
		pred_y = (output[:,y_idx] + offset_y.dimshuffle('x','x',0,1)).dimshuffle(0,'x',1,2,3)
		pred_w, pred_h = output[:,w_idx].dimshuffle(0,'x',1,2,3), output[:,h_idx].dimshuffle(0,'x',1,2,3)
		#pred_w, pred_h = T.exp(pred_w), T.exp(pred_h)		
		pred_conf = output[:,conf_idx].dimshuffle(0,'x',1,2,3)
		pred_class = output[:,-C:].dimshuffle(0,'x',1,2,3)
		
		#pred_w, pred_h = T.maximum(pred_w, 0.), T.maximum(pred_h, 0.)

		x_idx, y_idx = T.arange(0,truth.shape[1],4+C), T.arange(1,truth.shape[1],4+C)
		w_idx, h_idx = T.arange(2,truth.shape[1],4+C), T.arange(3,truth.shape[1],4+C)
		class_idx,_ = theano.scan(
			lambda x: T.arange(x,x+C,1),
			sequences = T.arange(4,truth.shape[1],4+C)
		)

		truth_x, truth_y = truth[:,x_idx], truth[:,y_idx]
		truth_w, truth_h = truth[:,w_idx], truth[:,h_idx]
		truth_class = truth[:, class_idx]
		
		# Get intersection region bounding box coordinates
		xi = T.maximum(pred_x, truth_x.dimshuffle(0,1,'x','x','x'))
		xf = T.minimum(pred_x + pred_w, (truth_x + truth_w).dimshuffle(0,1,'x','x','x'))
		yi = T.maximum(pred_y, truth_y.dimshuffle(0,1,'x','x','x'))
		yf = T.minimum(pred_y + pred_h, (truth_y + truth_h).dimshuffle(0,1,'x','x','x'))
		w, h = T.maximum(xf - xi, 0.), T.maximum(yf - yi, 0.)

		# Calculate iou score for predicted boxes and truth
		isec = w * h
		union = (pred_w * pred_h) + (truth_w * truth_h).dimshuffle(0,1,'x','x','x') - isec
		iou = T.maximum(isec/union, 0.)

		# Calculate rmse for boxes which have 0 iou score
		squared_error = (pred_x - truth_x.dimshuffle(0,1,'x','x','x'))**2 + (pred_y - truth_y.dimshuffle(0,1,'x','x','x'))**2 + \
			(pred_h - truth_h.dimshuffle(0,1,'x','x','x'))**2 + (pred_h - truth_h.dimshuffle(0,1,'x','x','x'))**2

		# Get index matrix representing max along the 1st dimension for the iou score (reps 'responsible' box).
		maxval_idx, _ = meshgrid2D(T.arange(B), T.arange(truth.shape[0]))
		maxval_idx = maxval_idx.dimshuffle(0,'x',1,'x','x')
		maxval_idx = T.repeat(T.repeat(maxval_idx,S[0],3),S[1],4)

		# determine which box is responsible by giving box with highest iou score (if iou > 0) or smalles squared error.
		greater_iou = T.eq(maxval_idx, iou.argmax(axis=2).dimshuffle(0,1,'x',2,3))
		smaller_se = T.eq(maxval_idx, squared_error.argmin(axis=2).dimshuffle(0,1,'x',2,3))
		box_is_resp = T.switch(iou.max(axis=2, keepdims=True) > 0, greater_iou, smaller_se)
		
		# Get matrix for the width/height of each cell
		width, height = T.ones(S) / S[1], T.ones(S) / S[0]
		width, height = width.dimshuffle('x','x',0,1), height.dimshuffle('x','x',0,1)
		offset_x, offset_y = offset_x.dimshuffle('x','x',0,1), offset_y.dimshuffle('x','x',0,1)

		# Get bounding box for intersection between CELL and ground truth box.
		xi = T.maximum(offset_x, truth_x.dimshuffle(0,1,'x','x'))
		xf = T.minimum(offset_x + width, (truth_x + truth_w).dimshuffle(0,1,'x','x'))
		yi = T.maximum(offset_y, truth_y.dimshuffle(0,1,'x','x'))
		yf = T.minimum(offset_y + height, (truth_y + truth_h).dimshuffle(0,1,'x','x'))
		w, h = T.maximum(xf - xi, 0.), T.maximum(yf - yi, 0.)

		# Calculate iou score for the cell.
		isec = w * h
		union = (width * height) + (truth_w* truth_h).dimshuffle(0,1,'x','x') - isec
		iou_cell = T.maximum(isec/union, 0.).dimshuffle(0,1,'x',2,3) # * (np.prod(S)) # normalize the iou to make more sense
		
		maxval_idx, _ = meshgrid2D(T.arange(iou_cell.shape[1]), T.arange(iou_cell.shape[0]))
		maxval_idx = maxval_idx.dimshuffle(0,1,'x','x','x')
		maxval_idx = T.repeat(T.repeat(T.repeat(maxval_idx, B, 2), S[0], 3), S[1], 4)
		
		obj_for_cell = T.eq(maxval_idx, iou_cell.argmax(axis=1).dimshuffle(0,'x',1,2,3))
			
		# Get logical matrix representing minimum iou score for cell to be considered overlapping ground truth.
		cell_intersects = (iou_cell > iou_thresh)
		
		obj_in_cell_and_resp = T.bitwise_and(T.bitwise_and(cell_intersects, box_is_resp), obj_for_cell)
		conf_is_zero = T.bitwise_and(
			bitwise_not(T.bitwise_and(cell_intersects, box_is_resp)),
			obj_for_cell
		)
		conf_is_zero = conf_is_zero.sum(axis=1, keepdims=True)
		
		# repeat "cell overlaps" logical matrix for the number of classes.
		pred_class = T.repeat(pred_class, truth.shape[1] // (4 + C), axis=1)

		# repeat the ground truth for class probabilities for each cell.
		truth_class_rep = T.repeat(T.repeat(truth_class.dimshuffle(0,1,2,'x','x'), S[0], axis=3), S[1], axis=4)
		cell_intersects = T.repeat(cell_intersects, C, axis=2)

		if not rescore:
			iou = T.ones_like(iou)
		cost = T.sum((pred_conf - iou)[obj_in_cell_and_resp.nonzero()]**2) + \
			lmbda_noobj * T.sum((pred_conf[conf_is_zero.nonzero()])**2) + \
		 	lmbda_coord * T.sum((pred_x - truth_x.dimshuffle(0,1,'x','x','x'))[obj_in_cell_and_resp.nonzero()]**2) + \
		 	lmbda_coord * T.sum((pred_y - truth_y.dimshuffle(0,1,'x','x','x'))[obj_in_cell_and_resp.nonzero()]**2) + \
			lmbda_coord * T.sum((safe_sqrt(pred_w) - safe_sqrt(truth_w.dimshuffle(0,1,'x','x','x')))[obj_in_cell_and_resp.nonzero()]**2) + \
			lmbda_coord * T.sum((safe_sqrt(pred_h) - safe_sqrt(truth_h.dimshuffle(0,1,'x','x','x')))[obj_in_cell_and_resp.nonzero()]**2) + \
			lmbda_obj * T.sum(((pred_class - truth_class_rep)[cell_intersects.nonzero()])**2)

		cost /= T.maximum(1., truth.shape[0])
		return cost, [iou]

	'''
	Implement functions for the BaseLearningObject Class
	'''
	def get_weights(self):
		return [p.get_value() for p in self.params]

	def get_hyperparameters(self):
		self._hyperparameters = super(YoloObjectDetector, self).get_hyperparameters()
		curr_hyperparameters = self._hyperparameters[-1]
		curr_hyperparameters.extend({
			'S': list(self.S),
			'B': self.B,
			'num_classes': self.num_classes
			})
		return self._hyperparameters

	def get_architecture(self):
		architecture = {}
		for layer in self.network:
			architecture[layer] = self.network[layer].__str__()
		return architecture

	def load_model(self, weights):
		layers.set_all_param_values(self.network['output'], weights)

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
		lmbda_coord = self.settings.lmbda_coord
		lmbda_noobj = self.settings.lmbda_noobj
		lmbda_obj = self.settings.lmbda_obj
		rescore = self.settings.rescore
		
		if not hasattr(self, '_train_fn') or not hasattr(self, '_test_fn'):
			if not hasattr(self, 'target'):
				self.target = T.matrix('target')

			print_obj.println('Getting cost...\n')
			ti = time.time()
			cost, constants = self._get_cost(self.output, self.target, self.S, self.B, self.num_classes, rescore=rescore, lmbda_coord=lmbda_coord, lmbda_noobj=lmbda_noobj, lmbda_obj=lmbda_obj)
			cost_test, _ = self._get_cost(self.output_test, self.target, self.S, self.B, self.num_classes, rescore=rescore, lmbda_coord=lmbda_coord, lmbda_noobj=lmbda_noobj, lmbda_obj=lmbda_obj)
			
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

	@staticmethod
	def nms(output, S, B, C, thresh=.3, overlap=.2):
		obj_idx = range(4,output.shape[0] - C, 5)
		scores = output[obj_idx] * output[-C:].max(axis=0, keepdims=True)
		scores_flat = scores.flatten()
		above_thresh_idx = np.arange(scores_flat.size)[scores_flat > thresh]
		preds = []
		for i in range(above_thresh_idx.size):
			idx = np.unravel_index(above_thresh_idx[i], scores.shape)
			pred = np.copy(output[idx[0]:idx[0] + 4, idx[1], idx[2]])
			pred[0], pred[1] = pred[0] + np.float_(idx[2])/S[1], pred[1] + np.float_(idx[1])/S[0]
			pred = np.concatenate((pred, [scores[idx[0],idx[1],idx[2]], np.argmax(output[-C:,idx[1],idx[2]])]))
			#adj_wh = pred[[2,3]]  # adjust width and height since training adds an extra factor
			# adj_wh[adj_wh < 1] = 0.5 * adj_wh[adj_wh < 1]**2
			#adj_wh[adj_wh >= 1] = np.abs(adj_wh[adj_wh >= 1]) - 0.5)
			#pred[[2,3]] = np.exp(adj_wh)
			#pred[[2,3]] += pred[[0,1]] # turn width and height into xf, yf
			preds.append(pred)
		preds = np.asarray(preds)
		return preds
		if preds.shape[0] == 0:
			return np.zeros((0,6))
		
		nms_preds = np.zeros((0,6))
		for cls in range(C):
			idx = preds[:,-1] == cls
			cls_preds = preds[idx]
			cls_preds = utils.nms(cls_preds, overlap)
			if cls_preds.shape[0] > 0:
				cls_preds[:,[2,3]] -= cls_preds[:,[0,1]]
			nms_preds = np.concatenate((nms_preds, cls_preds), axis=0)
		return nms_preds

