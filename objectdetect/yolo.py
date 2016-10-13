import theano
from theano import tensor as T
import numpy as np
from bnr_ml.nnet.updates import momentum as momentum_update
from bnr_ml.nnet.layers import AbstractNNetLayer
from bnr_ml.utils.helpers import meshgrid2D
from collections import OrderedDict
from tqdm import tqdm
import time
from lasagne import layers
from lasagne.updates import rmsprop

import pdb

class YoloObjectDetectorError(Exception):
		pass


class YoloObjectDetector(object):
	'''

	'''
	def __init__(
		self,
		network,
		input_shape,
		num_classes,
		S,
		B):
		'''
		network:
		--------
			Dict with the entire network defined, must have a "feature_map" and "output" layer.
			You must be able to call .get_output() on these layers.
		'''
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
			for i in range(num_classes):
				output = T.set_subtensor(output[:,5*i + 2:5*i + 4,:,:], T.nnet.sigmoid(output[:,5*i + 2:5*i + 4,:,:]))
				output = T.set_subtensor(output[:,5*i + 4,:,:], T.nnet.sigmoid(output[:,5*i + 4,:,:]))
			output = T.set_subtensor(output[:,-self.num_classes:,:,:], T.exp(output[:,-self.num_classes:,:,:]) / T.sum(T.exp(output[:,-self.num_classes:,:,:]), axis=1, keepdims=True))
			return output
		self.output = get_output(output, B, S, num_classes)
		self.output_test = get_output(output_test, B, S, num_classes)

		self.params = layers.get_all_params(network['output'])
		# self.params = []
		# for lname in network:
		# 	layer = network[lname]
		# 	self.params.extend(layer.get_params())

	def _get_cost_optim(self, output, truth, S, B, C, lmbda_coord=5., lmbda_noobj=0.5, iou_thresh=0.1):
		# calculate height/width of individual cell
		block_height, block_width = 1. / S[0], 1./ S[1]

		# get the offset of each cell
		offset_x, offset_y = meshgrid2D(T.arange(0,1,block_height), T.arange(0,1,block_width))
		
		# get indices for x,y,w,h,object-ness for easy access
		x_idx, y_idx = T.arange(0,5*B,5), T.arange(1,5*B, 5)
		w_idx, h_idx = T.arange(2,5*B,5), T.arange(3,5*B,5)
		conf_idx = T.arange(4,5*B+4,5)
		
		# Get position predictions with offsets.
		pred_x = output[:,x_idx] + offset_x.dimshuffle('x','x',0,1)
		pred_y = output[:,y_idx] + offset_y.dimshuffle('x','x',0,1)
		pred_w, pred_h, pred_conf = output[:,w_idx], output[:,h_idx], output[:,conf_idx]
		
		# Get intersection region bounding box coordinates
		xi = T.maximum(pred_x, truth[:,0].dimshuffle(0,'x','x','x'))
		xf = T.minimum(pred_x + pred_w, (truth[:,0] + truth[:,2]).dimshuffle(0,'x','x','x'))
		yi = T.maximum(pred_y, truth[:,1].dimshuffle(0,'x','x','x'))
		yf = T.minimum(pred_y + pred_h, (truth[:,1] + truth[:,3]).dimshuffle(0,'x','x','x'))
		
		# Calculate iou score for predicted boxes and truth
		isec = (xf - xi) * (yf - yi)
		union = (pred_w * pred_h) + (truth[:,2] * truth[:,3]).dimshuffle(0,'x','x','x') - isec
		iou = T.maximum(isec/union, 0.)

		# Get index matrix representing max along the 1st dimension for the iou score (reps 'responsible' box).
		maxval_idx, a = meshgrid2D(T.arange(B), T.arange(truth.shape[0]))
		maxval_idx = maxval_idx.reshape((truth.shape[0],-1))
		maxval_idx = maxval_idx.dimshuffle(0,1,'x','x')
		maxval_idx = T.repeat(T.repeat(maxval_idx,S[0],2),S[1],3)
		is_max = T.eq(maxval_idx, iou.argmax(axis=1).dimshuffle(0,'x',1,2))
		
		# Get matrix for the width/height of each cell
		width, height = T.ones(S) / S[1], T.ones(S) / S[0]
		width, height = width.dimshuffle('x',0,1), height.dimshuffle('x',0,1)
		
		# Get bounding box for intersection between CELL and ground truth box.
		xi = T.maximum(offset_x, truth[:,0].dimshuffle(0,'x','x'))
		xf = T.minimum(offset_x + width, (truth[:,0] + truth[:,2]).dimshuffle(0,'x','x'))
		yi = T.maximum(offset_y, truth[:,1].dimshuffle(0,'x','x'))
		yf = T.minimum(offset_y + height, (truth[:,1] + truth[:,3]).dimshuffle(0,'x','x'))

		# Calculate iou score for the cell.
		isec = (xf - xi) * (yf - yi)
		union = width * height + (truth[:,2] * truth[:,3]).dimshuffle(0,'x','x') - isec
		iou_cell = T.maximum(isec/union, 0.)
		
		# Get logical matrix representing minimum iou score for cell to be considered overlapping ground truth.
		is_inter = (iou_cell > iou_thresh).dimshuffle(0,'x',1,2)
		
		# Get logical matrix for cells and boxes which overlap and are responsible for prediction.
		isbox_andcell = T.bitwise_and(is_max, is_inter)
		
		# repeat "cell overlaps" logical matrix for the number of classes.
		is_inter = T.repeat(is_inter, C, axis=1)
		
		# repeat the ground truth for class probabilities for each cell.
		clspred_truth = T.repeat(T.repeat(truth[:,-C:].dimshuffle(0,1,'x','x'), S[0], axis=2), S[1], axis=3)
		
		# calculate cost
		cost = lmbda_coord * T.sum((pred_conf - iou)[isbox_andcell.nonzero()]**2) + \
			lmbda_noobj * T.sum((pred_conf[T.bitwise_not(isbox_andcell).nonzero()])**2) + \
			T.sum((pred_x[is_max.nonzero()].reshape((truth.shape[0],-1)) - truth[:,[0]])**2) + \
			T.sum((pred_y[is_max.nonzero()].reshape((truth.shape[0],-1)) - truth[:,[1]])**2) + \
			T.sum((pred_w[is_max.nonzero()].reshape((truth.shape[0],-1)).sqrt() - truth[:,[2]].sqrt())**2) + \
			T.sum((pred_h[is_max.nonzero()].reshape((truth.shape[0],-1)).sqrt() - truth[:,[3]].sqrt())**2) + \
			T.sum((output[:,-C:][is_inter.nonzero()] - clspred_truth[is_inter.nonzero()])**2)
		
		return cost

	def _get_cost(self, output, target, lmbda_coord=10., lmbda_noobj = .1, iou_thresh = .1):
		lmbda_coord = T.as_tensor_variable(lmbda_coord)
		lmbda_noobj = T.as_tensor_variable(lmbda_noobj)
		iou_thresh = T.as_tensor_variable(iou_thresh)
		# output = self.output
		#if isinstance(output, AbstractNNetLayer):
		#	output = output.get_output()
		dims, probs = target[:,:4], target[:,4:]

		def scale_dims(dims, i, j):
			dims = T.set_subtensor(dims[:,0], dims[:,0] - float(i) / self.S[0])
			dims = T.set_subtensor(dims[:,1], dims[:,1] - float(j) / self.S[1])
			return dims
		def unscale_dims(dims, i, j):
			dims = T.set_subtensor(dims[:,0], dims[:,0] + float(i) / self.S[0])
			dims = T.set_subtensor(dims[:,1], dims[:,1] + float(j) / self.S[1])
			return dims
		def iou_score(box1, box2, eps=1e-3):
			xi = T.maximum(box1[:,0], box2[:,0])
			yi = T.maximum(box1[:,1], box2[:,1])
			xf = T.minimum(box1[:,0] + box1[:,2], box2[:,0] + box2[:,2])
			yf = T.minimum(box1[:,1] + box1[:,3], box2[:,1] + box2[:,3])

			isec = T.maximum((xf - xi) * (yf - yi), 0.)
			union = box1[:,2]*box1[:,3] + box2[:,2]*box2[:,3] - isec
			return isec / (union + eps)

		cost = T.as_tensor_variable(0.)
		for i in range(self.S[0]):
			for j in range(self.S[1]):
				preds_ij = []
				box_ious = []

				dims = scale_dims(dims, i, j)

				for k in range(self.B):
					pred_ijk = output[:,k*5:(k+1)*5,i,j] # single prediction for cell and box

					# calc iou score
					iou = iou_score(dims, pred_ijk)

					# append to iou list (iou scores for each box B)
					preds_ij.append(pred_ijk.dimshuffle(0,1,'x'))
					box_ious.append(iou.dimshuffle(0,'x'))

				# Determine if the image intersects with the cell
				iou = iou_score(dims, np.asarray([[0., 0., 1./self.S[0], 1./self.S[1]]]))

				is_not_in_cell = (iou < iou_thresh).nonzero()

				preds_ij = T.concatenate(preds_ij, axis=2)
				box_ious = T.concatenate(box_ious, axis=1)

				iou_max = T.argmax(box_ious, axis=1)

				# get final values for predictions
				row,col = meshgrid2D(T.arange(preds_ij.shape[0]), T.arange(preds_ij.shape[1]))
				dep,col = meshgrid2D(iou_max, T.arange(preds_ij.shape[1]))

				preds_ij = preds_ij[row,col,dep].reshape(preds_ij.shape[:2])

				# get final values for IoUs
				row = T.arange(preds_ij.shape[0])
				box_ious = box_ious[row, iou_max]

				# is_not_responsible = (box_ious < iou_thresh).nonzero()
				box_ious = T.set_subtensor(box_ious[is_not_in_cell], 0.)

				cost_ij_t1 = (preds_ij[:,0] - dims[:,0])**2 + (preds_ij[:,1] - dims[:,1])**2
				cost_ij_t1 +=  (T.sqrt(preds_ij[:,2]) - T.sqrt(dims[:,2]))**2 + (T.sqrt(preds_ij[:,3]) - T.sqrt(dims[:,3]))**2

				conf_in_cell = T.set_subtensor(preds_ij[:,4][is_not_in_cell], 0.) # set values which don't intersect with cell to 0.

				cost_ij_t1 += 0*(conf_in_cell - box_ious)**2

				cost_ij_t1 *= lmbda_coord
				
				conf_out_cell = T.set_subtensor(preds_ij[:,4][T.invert(is_not_in_cell)], 0.) # set values which do interset with cell to 0.
				cost_ij_t1 += 0 * lmbda_noobj * (conf_out_cell - 0.)**2 # set non-intersecting cell confidences to 0.

				cost_ij_t2 = lmbda_noobj * T.sum((probs - output[:,-self.num_classes:,i,j])**2, axis=1)

				#cost_ij_t1 = T.set_subtensor(cost_ij_t1[is_box_not_in_cell], 0.)
				cost_ij_t2 = T.set_subtensor(cost_ij_t2[is_not_in_cell], 0.)

				cost += cost_ij_t1 + cost_ij_t2

				dims = unscale_dims(dims, i, j)

		cost = cost.mean()

		return cost

	def _get_updates(self, cost, params, lr=1e-4):
		lr = T.as_tensor_variable(lr)
		updates = OrderedDict()
		grads = T.grad(cost, params)
		for param, grad in zip(params, grads):
			updates[param] = param - lr * grad

		return updates

	def train(self, X, y, batch_size=50, epochs=10, train_test_split=0.8, lr=1e-4, momentum=0.9, lmbda_coord=5., lmbda_noobj=0.5, target=None, seed=1991):
		np.random.seed(seed)
		
		if target is None:
			target = T.matrix('target')

		print('Getting cost...'); time.sleep(0.1)
		ti = time.time()
		cost = self._get_cost_optim(self.output, target, self.S, self.B, self.num_classes, lmbda_coord=lmbda_coord, lmbda_noobj=lmbda_noobj)
		cost_test = self._get_cost_optim(self.output_test, target, self.S, self.B, self.num_classes, lmbda_coord=lmbda_coord, lmbda_noobj=lmbda_noobj)
		print("Creating cost variable took %.4f seconds" % (time.time() - ti,))

		#updates = momentum_update(cost, self.params, lr=lr, momentum=momentum)
		updates = rmsprop(cost, self.params, learning_rate=lr)

		print('Compiling...'); time.sleep(0.1)
		ti = time.time()
		train_fn = theano.function([self.input, target], cost, updates=updates)
		test_fn = theano.function([self.input, target], cost_test)
		print("Compiling functions took %.4f seconds" % (time.time() - ti,))

		Ntrain = np.int_(X.shape[0] * train_test_split)

		Xtrain, ytrain = X[:Ntrain], y[:Ntrain]
		Xtest, ytest = X[Ntrain:], y[Ntrain:]

		train_loss = np.zeros((epochs,))
		test_loss = np.zeros((epochs,))

		print('Beginning training...'); time.sleep(0.1)
		for epoch in tqdm(range(epochs)):
			idx = np.arange(Xtrain.shape[0])
			np.random.shuffle(idx)
			Xtrain, ytrain = Xtrain[idx], ytrain[idx]

			train_loss_batch = []

			for i in range(0, Xtrain.shape[0], batch_size):
				Xbatch, ybatch = Xtrain[i:i + batch_size], ytrain[i:i + batch_size]
				if Xbatch.shape[0] > 0:
					err = train_fn(Xbatch, ybatch)
					train_loss_batch.append(err)
			train_loss[epoch] = np.mean(train_loss_batch)
			test_loss[epoch] = test_fn(Xtest, ytest)

			print('Epoch %d\n------\nTrain Loss: %.4f, Test Loss: %.4f' % (epoch, train_loss[epoch], test_loss[epoch])); time.sleep(0.1)	

		return train_loss, test_loss
















