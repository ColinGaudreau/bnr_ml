import theano
from theano import tensor as T
import numpy as np
from bnr_ml.nnet.updates import momentum as momentum_update
from bnr_ml.nnet.layers import AbstractNNetLayer
from bnr_ml.utils.helpers import meshgrid2D, bitwise_not
from bnr_ml.utils.nonlinearities import softmax, smooth_l1, safe_sqrt
from collections import OrderedDict
from tqdm import tqdm
import time

from lasagne import layers
from lasagne.updates import rmsprop, sgd

from itertools import tee

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
			for i in range(B):
				#output = T.set_subtensor(output[:,5*i:5*i+2,:,:], 2 * T.nnet.sigmoid(output[:,5*i:5*i+2,:,:]) - 1)
				#output = T.set_subtensor(output[:,5*i + 2:5*i + 4,:,:], T.nnet.sigmoid(output[:,5*i + 2:5*i + 4,:,:]))
				#output = T.set_subtensor(output[:,5*i + 4,:,:], T.nnet.sigmoid(output[:,5*i + 4,:,:]))
				pass
			output = T.set_subtensor(output[:,-self.num_classes:,:,:], softmax(output[:,-self.num_classes:,:,:], axis=1)) # use safe softmax
			return output
		self.output = get_output(output, B, S, num_classes)
		self.output_test = get_output(output_test, B, S, num_classes)

		self.params = layers.get_all_params(network['output'])

	def _get_cost_optim(self, output, truth, S, B, C, lmbda_coord=5., lmbda_noobj=0.5, iou_thresh=0.05):
		# calculate height/width of individual cell
		block_height, block_width = 1. / S[0], 1./ S[1]

		# get the offset of each cell
		offset_x, offset_y = meshgrid2D(T.arange(0,1,block_width), T.arange(0,1,block_height))

		# get indices for x,y,w,h,object-ness for easy access
		x_idx, y_idx = T.arange(0,5*B,5), T.arange(1,5*B, 5)
		w_idx, h_idx = T.arange(2,5*B,5), T.arange(3,5*B,5)
		conf_idx = T.arange(4,5*B,5)
		
		# Get position predictions with offsets.
		pred_x = output[:,x_idx] + offset_x.dimshuffle('x','x',0,1)
		pred_y = output[:,y_idx] + offset_y.dimshuffle('x','x',0,1)
		pred_w, pred_h, pred_conf = output[:,w_idx], output[:,h_idx], output[:,conf_idx]
		pred_w, pred_h = T.maximum(pred_w, 0.), T.maximum(pred_h, 0.)

		truth_x, truth_y, truth_w, truth_h = truth[:,0], truth[:,1], truth[:,2], truth[:,3]
		truth_w, truth_h = T.maximum(truth_w, 0.), T.maximum(truth_h, 0.)
		
		# Get intersection region bounding box coordinates
		xi = T.maximum(pred_x, truth_x.dimshuffle(0,'x','x','x'))
		xf = T.minimum(pred_x + pred_w, (truth_x + truth_w).dimshuffle(0,'x','x','x'))
		yi = T.maximum(pred_y, truth_y.dimshuffle(0,'x','x','x'))
		yf = T.minimum(pred_y + pred_h, (truth_y + truth_h).dimshuffle(0,'x','x','x'))
		w, h = T.maximum(xf - xi, 0.), T.maximum(yf - yi, 0.)

		# Calculate iou score for predicted boxes and truth
		isec = w * h
		union = (pred_w * pred_h) + (truth_w * truth_h).dimshuffle(0,'x','x','x') - isec
		iou = T.maximum(isec/union, 0.)

		# Get index matrix representing max along the 1st dimension for the iou score (reps 'responsible' box).
		maxval_idx, _ = meshgrid2D(T.arange(B), T.arange(truth.shape[0]))
		maxval_idx = maxval_idx.dimshuffle(0,1,'x','x')
		maxval_idx = T.repeat(T.repeat(maxval_idx,S[0],2),S[1],3)
		
		is_max = T.eq(maxval_idx, iou.argmax(axis=1).dimshuffle(0,'x',1,2))
		is_not_max = T.neq(maxval_idx, iou.argmax(axis=1).dimshuffle(0,'x',1,2))
		
		# Get matrix for the width/height of each cell
		width, height = T.ones(S) / S[1], T.ones(S) / S[0]
		width, height = width.dimshuffle('x',0,1), height.dimshuffle('x',0,1)
		offset_x, offset_y = offset_x.dimshuffle('x',0,1), offset_y.dimshuffle('x',0,1)
		
		# Get bounding box for intersection between CELL and ground truth box.
		xi = T.maximum(offset_x, truth_x.dimshuffle(0,'x','x'))
		xf = T.minimum(offset_x + width, (truth_x + truth_w).dimshuffle(0,'x','x'))
		yi = T.maximum(offset_y, truth_y.dimshuffle(0,'x','x'))
		yf = T.minimum(offset_y + height, (truth_y + truth_h).dimshuffle(0,'x','x'))
		w, h = T.maximum(xf - xi, 0.), T.maximum(yf - yi, 0.)

		# Calculate iou score for the cell.
		isec = (xf - xi) * (yf - yi)
		union = (width * height) + (truth_w* truth_h).dimshuffle(0,'x','x') - isec
		iou_cell = T.maximum(isec/union, 0.)
		
		# Get logical matrix representing minimum iou score for cell to be considered overlapping ground truth.
		is_inter = (iou_cell > iou_thresh).dimshuffle(0,'x',1,2)

		obj_in_cell_and_resp = T.bitwise_and(is_inter, is_max)
		
		# repeat "cell overlaps" logical matrix for the number of classes.
		is_inter = T.repeat(is_inter, C, axis=1)
		
		# repeat the ground truth for class probabilities for each cell.
		clspred_truth = T.repeat(T.repeat(truth[:,-C:].dimshuffle(0,1,'x','x'), S[0], axis=2), S[1], axis=3)
		
		# calculate cost
		cost = T.sum((pred_conf - iou)[obj_in_cell_and_resp.nonzero()]**2) + \
			lmbda_noobj * T.sum((pred_conf[bitwise_not(obj_in_cell_and_resp).nonzero()])**2) + \
			lmbda_coord * T.sum((pred_x - truth[:,0].dimshuffle(0,'x','x','x'))[obj_in_cell_and_resp.nonzero()]**2) + \
			lmbda_coord * T.sum((pred_y - truth[:,1].dimshuffle(0,'x','x','x'))[obj_in_cell_and_resp.nonzero()]**2) + \
			lmbda_coord * T.sum((safe_sqrt(pred_w) - truth_w.dimshuffle(0,'x','x','x').sqrt())[obj_in_cell_and_resp.nonzero()]**2) + \
			lmbda_coord * T.sum((safe_sqrt(pred_h) - truth_h.dimshuffle(0,'x','x','x').sqrt())[obj_in_cell_and_resp.nonzero()]**2) + \
			T.sum((output[:,-C:][is_inter.nonzero()] - clspred_truth[is_inter.nonzero()])**2)
		
		return cost / T.maximum(1., truth.shape[0])

	def _get_cost_optim_multi(self, output, truth, S, B, C,lmbda_coord=5., lmbda_noobj=0.5, lmbda_obj=1., iou_thresh=1e-3):
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
		pred_w, pred_h = smooth_l1(pred_w), smooth_l1(pred_h)		
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

		# Get index matrix representing max along the 1st dimension for the iou score (reps 'responsible' box).
		maxval_idx, _ = meshgrid2D(T.arange(B), T.arange(truth.shape[0]))
		maxval_idx = maxval_idx.dimshuffle(0,'x',1,'x','x')
		maxval_idx = T.repeat(T.repeat(maxval_idx,S[0],3),S[1],4)

		box_is_resp = T.eq(maxval_idx, iou.argmax(axis=2).dimshuffle(0,1,'x',2,3))

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
	
		cost = T.sum((pred_conf - iou)[obj_in_cell_and_resp.nonzero()]**2) + \
			lmbda_noobj * T.sum((pred_conf[conf_is_zero.nonzero()])**2) + \
		 	lmbda_coord * T.sum((pred_x - truth_x.dimshuffle(0,1,'x','x','x'))[obj_in_cell_and_resp.nonzero()]**2) + \
		 	lmbda_coord * T.sum((pred_y - truth_y.dimshuffle(0,1,'x','x','x'))[obj_in_cell_and_resp.nonzero()]**2) + \
			lmbda_coord * T.sum((safe_sqrt(pred_w) - safe_sqrt(truth_w).dimshuffle(0,1,'x','x','x'))[obj_in_cell_and_resp.nonzero()]**2) + \
			lmbda_coord * T.sum((safe_sqrt(pred_h) - safe_sqrt(truth_h).dimshuffle(0,1,'x','x','x'))[obj_in_cell_and_resp.nonzero()]**2) + \
			lmbda_obj * T.sum(((pred_class - truth_class_rep)[cell_intersects.nonzero()])**2)

		cost /= T.maximum(1., truth.shape[0])
		pdb.set_trace()	
		return cost, [iou, obj_in_cell_and_resp, conf_is_zero, obj_in_cell_and_resp, cell_intersects]

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

	def train(self, train_gen, test_gen, batch_size=50, epochs=10, train_test_split=0.8, lr=1e-4, momentum=0.9, lmbda_coord=5., lmbda_noobj=0.5, target=None, seed=1991, logfile='/dev/stdout'):
		np.random.seed(seed)
		
		logfile = open(logfile, 'w')

		if target is None:
			target = T.matrix('target')
		
		logfile.write('Getting cost...\n')
		print('Getting cost...'); time.sleep(0.1)
		ti = time.time()
		cost, constants = self._get_cost_optim_multi(self.output, target, self.S, self.B, self.num_classes, lmbda_coord=lmbda_coord, lmbda_noobj=lmbda_noobj)
		cost_test, _ = self._get_cost_optim_multi(self.output_test, target, self.S, self.B, self.num_classes, lmbda_coord=lmbda_coord, lmbda_noobj=lmbda_noobj)
		
		logfile.write("Creating cost variable took %.4f seconds\n" % (time.time() - ti,))
		print("Creating cost variable took %.4f seconds" % (time.time() - ti,))

		#updates = momentum_update(cost, self.params, lr=lr, momentum=momentum)
		grads = T.grad(cost, self.params, consider_constant=constants)
		updates = rmsprop(grads, self.params, learning_rate=lr)
		#updates = sgd(grads, self.params, learning_rate=lr)
		
		logfile.write('Compiling...\n')
		print('Compiling...'); time.sleep(0.1)
		ti = time.time()
		train_fn = theano.function([self.input, target], cost, updates=updates)
		test_fn = theano.function([self.input, target], cost_test)
		
		logfile.write('Compiling functions took %.4f seconds\n' % (time.time() - ti,))
		print("Compiling functions took %.4f seconds" % (time.time() - ti,))

		# Ntrain = np.int_(X.shape[0] * train_test_split)

		# Xtrain, ytrain = X[:Ntrain], y[:Ntrain]
		# Xtest, ytest = X[Ntrain:], y[Ntrain:]

		train_loss = np.zeros((epochs,))
		test_loss = np.zeros((epochs,))

		logfile.write('Beginning training...\n')
		print('Beginning training...'); time.sleep(0.1)

		try:
			for epoch in tqdm(range(epochs)):
				#idx = np.arange(Xtrain.shape[0])
				#np.random.shuffle(idx)
				#Xtrain, ytrain = Xtrain[idx], ytrain[idx]

				train_loss_batch = []
				test_loss_batch = []

				train_gen, train_gen_backup = tee(train_gen)
				test_gen, test_gen_backup = tee(test_gen)

				for Xbatch, ybatch in train_gen:
					err = train_fn(Xbatch, ybatch)
					logfile.write('Batch error: %.4f\n' % err)
					print(err)
					train_loss_batch.append(err)

				for Xbatch, ybatch in test_gen:
					test_loss_batch.append(test_fn(Xbatch, ybatch))

				train_loss[epoch] = np.mean(train_loss_batch)
				test_loss[epoch] = np.mean(test_loss_batch)

				train_gen = train_gen_backup
				test_gen = test_gen_backup
				
				logfile.write('Epoch %d\n------\nTrain Loss: %.4f, Test Loss: %.4f\n' % (epoch, train_loss[epoch], test_loss[epoch]))
				print('Epoch %d\n------\nTrain Loss: %.4f, Test Loss: %.4f' % (epoch, train_loss[epoch], test_loss[epoch])); time.sleep(0.1)
		except KeyboardInterrupt:
			logfile.close()
		
		logfile.close()
		return train_loss, test_loss
















