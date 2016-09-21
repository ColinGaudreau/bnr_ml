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

		output = layers.get_output(network['output'])
		output = T.reshape(output, (-1, B * 5 + num_classes, S[0], S[1]))
		for i in range(num_classes):
			output = T.set_subtensor(output[:,5*i + 2:5*i + 4,:,:], T.nnet.sigmoid(output[:,5*i + 2:5*i + 4,:,:]))
			output = T.set_subtensor(output[:,5*i + 4,:,:], T.nnet.sigmoid(output[:,5*i + 4,:,:]))
		output = T.set_subtensor(output[:,-self.num_classes:,:,:], T.exp(output[:,-self.num_classes:,:,:]) / T.sum(T.exp(output[:,-self.num_classes:,:,:]), axis=1, keepdims=True))
		self.output = output

		self.params = layers.get_all_params(network['output'])
		# self.params = []
		# for lname in network:
		# 	layer = network[lname]
		# 	self.params.extend(layer.get_params())

	def _get_cost(self, target, lmbda_coord=10., lmbda_noobj = .1, iou_thresh = .1):
		lmbda_coord = T.as_tensor_variable(lmbda_coord)
		lmbda_noobj = T.as_tensor_variable(lmbda_noobj)
		iou_thresh = T.as_tensor_variable(iou_thresh)
		output = self.output
		#if isinstance(output, AbstractNNetLayer):
		#	output = output.get_output()
		dims, probs = target[:,:4], target[:,4:]

		w1, w2 = np.ceil(float(self.input_shape[2]) / self.S[0]), np.ceil(float(self.input_shape[3]) / self.S[1])

		def scale_dims(dims):
			newdims = T.set_subtensor(dims[:,0], (dims[:,0] - i * w1) / self.input_shape[2])
			newdims = T.set_subtensor(newdims[:,1], (newdims[:,1] - j * w2) / self.input_shape[3])
			newdims = T.set_subtensor(newdims[:,2], (newdims[:,2] / self.input_shape[2]))
			newdims = T.set_subtensor(newdims[:,3], (newdims[:,3] / self.input_shape[3]))
			return newdims
		def unscale_dims(dims):
			newdims = T.set_subtensor(dims[:,0], dims[:,0] * self.input_shape[2] + i * w1)
			newdims = T.set_subtensor(newdims[:,1], newdims[:,1] * self.input_shape[3] + j * w2)
			newdims = T.set_subtensor(newdims[:,2], newdims[:,2] * self.input_shape[2])
			newdims = T.set_subtensor(newdims[:,3], newdims[:,3] * self.input_shape[3])
			return newdims

		cost = T.as_tensor_variable(0.)
		for i in range(self.S[0]):
			for j in range(self.S[1]):
				preds_ij = []
				ious = []

				newdims = scale_dims(dims)

				for k in range(self.B):
					pred_ijk = output[:,k*5:(k+1)*5,i,j] # single prediction for cell and box

					# get intersecion box coordinates relative to boxes
					isec_xi = T.maximum(newdims[:,0], pred_ijk[:,0])
					isec_yi = T.maximum(newdims[:,1], pred_ijk[:,1])
					isec_xf = T.minimum(newdims[:,0] + newdims[:,2], pred_ijk[:,0] + pred_ijk[:,2])
					isec_yf = T.minimum(newdims[:,1] + newdims[:,3], pred_ijk[:,1] + pred_ijk[:,3])

					isec = T.maximum((isec_xf - isec_xi) * (isec_yf - isec_yi), 0.)

					union = newdims[:,2] * newdims[:,3] + pred_ijk[:,2] * pred_ijk[:,3] - isec

					iou = isec / (union + T.as_tensor_variable(1e-2))

					preds_ij.append(pred_ijk.dimshuffle(0,1,'x'))
					ious.append(iou.dimshuffle(0,'x'))

				# Determine if the image intersects with the cell
				isec_xi = T.maximum(newdims[:,0], 0.)
				isec_yi = T.maximum(newdims[:,1], 0.)
				isec_xf = T.minimum(newdims[:,0] + newdims[:,2], 1. / self.S[0])
				isec_yf = T.minimum(newdims[:,1] + newdims[:,3], 1. / self.S[1])

				isec = T.maximum((isec_xf - isec_xi) * (isec_yf - isec_yi), 0.)

				union = newdims[:,2] * newdims[:,3] + pred_ijk[:,2] * pred_ijk[:,3] - isec

				iou = isec / (union + T.as_tensor_variable(1e-2))

				is_not_in_cell = (iou < iou_thresh).nonzero()

				preds_ij = T.concatenate(preds_ij, axis=2)
				ious = T.concatenate(ious, axis=1)

				iou_max = T.argmax(ious, axis=1)

				# get final values for predictions
				row,col = meshgrid2D(T.arange(preds_ij.shape[0]), T.arange(preds_ij.shape[1]))
				dep,col = meshgrid2D(iou_max, T.arange(preds_ij.shape[1]))

				preds_ij = preds_ij[row,col,dep].reshape(preds_ij.shape[:2])

				# get final values for IoUs
				row = T.arange(preds_ij.shape[0])
				ious = ious[row, iou_max]

				is_box_not_in_cell = (ious < iou_thresh).nonzero()

				cost_ij_t1 = (preds_ij[:,0] - newdims[:,0])**2 + (preds_ij[:,1] - newdims[:,1])**2
				cost_ij_t1 +=  (T.sqrt(preds_ij[:,2]) - T.sqrt(newdims[:,2]))**2 + (T.sqrt(preds_ij[:,3]) - T.sqrt(newdims[:,3]))**2
				cost_ij_t1 *= lmbda_coord

				cost_ij_t1 = lmbda_noobj * (preds_ij[:,4] - ious)**2

				cost_ij_t2 = lmbda_noobj * T.sum((probs - output[:,-self.num_classes:,i,j])**2, axis=1)

				cost_ij_t1 = T.set_subtensor(cost_ij_t1[is_box_not_in_cell], 0.)
				cost_ij_t2 = T.set_subtensor(cost_ij_t2[is_not_in_cell], 0.)

				cost += cost_ij_t1 + cost_ij_t2

				dims = unscale_dims(newdims)

		cost = cost.mean()

		return cost

	def _get_updates(self, cost, params, lr=1e-4):
		lr = T.as_tensor_variable(lr)
		updates = OrderedDict()
		grads = T.grad(cost, params)
		for param, grad in zip(params, grads):
			updates[param] = param - lr * grad

		return updates

	def train(self, X, y, batch_size=50, epochs=10, train_test_split=0.8, lr=1e-4, momentum=0.9, seed=1991):
		np.random.seed(seed)

		target = T.matrix('target')

		print('Getting cost...'); time.sleep(0.1)
		cost = self._get_cost(target)

		#updates = momentum_update(cost, self.params, lr=lr, momentum=momentum)
		updates = rmsprop(cost, self.params, learning_rate=lr)

		print('Compiling...'); time.sleep(0.1)
		train_fn = theano.function([self.input, target], cost, updates=updates)
		test_fn = theano.function([self.input, target], cost)

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
			test_loss[epoch] = test_fn(Xtrain, ytrain)

			print('Epoch %d\n------\nTrain Loss: %.4f, Test Loss: %.4f' % (epoch, train_loss[epoch], test_loss[epoch])); time.sleep(0.1)	

		return train_loss, test_loss
















