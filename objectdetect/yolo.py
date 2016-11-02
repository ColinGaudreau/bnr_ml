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
from PIL import Image, ImageDraw

from lasagne import layers
from lasagne.updates import rmsprop, sgd

from itertools import tee

import pdb

class YoloObjectDetector(object):
	'''

	'''
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

	def _get_cost__outdated(self, output, truth, S, B, C, lmbda_coord=5., lmbda_noobj=0.5, iou_thresh=0.05):
		'''
			Takes only one annotation, this limits training to one object per image (which is bad).
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

	def _get_cost(self, output, truth, S, B, C,lmbda_coord=5., lmbda_noobj=0.5, lmbda_obj=1., iou_thresh=1e-3):
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

	def _get_updates(self, cost, params, lr=1e-4):
		lr = T.as_tensor_variable(lr)
		updates = OrderedDict()
		grads = T.grad(cost, params)
		for param, grad in zip(params, grads):
			updates[param] = param - lr * grad

		return updates

	def train(
			self,
			train_gen,
			test_gen,
			epochs=10,
			lr=1e-4,
			momentum=0.9,
			lmbda_coord=5.,
			lmbda_noobj=0.5,
			target=None,
			seed=1991, 
			logfile='/dev/stdout'
		):
		np.random.seed(seed)
		
		logfile = open(logfile, 'w')

		if target is None:
			target = T.matrix('target')
		
		logfile.write('Getting cost...\n')
		print('Getting cost...'); time.sleep(0.1)
		ti = time.time()
		cost, constants = self._get_cost(self.output, target, self.S, self.B, self.num_classes, lmbda_coord=lmbda_coord, lmbda_noobj=lmbda_noobj)
		cost_test, _ = self._get_cost(self.output_test, target, self.S, self.B, self.num_classes, lmbda_coord=lmbda_coord, lmbda_noobj=lmbda_noobj)
		
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
			adj_wh = pred[[2,3]]  # adjust width and height since training adds an extra factor
			adj_wh[adj_wh < 1] = 0.5 * adj_wh[adj_wh < 1]**2
			adj_wh[adj_wh >= 1] = np.abs(adj_wh[adj_wh >= 1]) - 0.5
			pred[[2,3]] = adj_wh
			pred[[2,3]] += pred[[0,1]] # turn width and height into xf, yf
			preds.append(pred)
		preds = np.asarray(preds)
		
		if preds.shape[0] == 0:
			return np.zeros((0,6))
		
		def _nms(preds, thresh):
			if preds.shape[0] == 0:
				return preds
			idx = np.argsort(preds[:,4])
			pick = np.zeros_like(idx).astype(np.int32)
			area = np.maximum(0, (preds[:,2] - preds[:,0])) * np.maximum(0, (preds[:,3] - preds[:,1]))
			counter = 0
			while idx.size > 0:
				last = idx.size - 1
				i = idx[last]
				pick[counter] = i
				counter = counter + 1

				xi = np.maximum(preds[i,0], preds[idx[0:last - 1],0])
				xf = np.minimum(preds[i,2], preds[idx[0:last - 1],2])
				yi = np.maximum(preds[i,1], preds[idx[0:last - 1],1])
				yf = np.minimum(preds[i,3], preds[idx[0:last - 1],3])

				w, h = np.maximum(0., xf - xi), np.maximum(0., yf - yi)
				isec = w * h
				o = isec / (area[idx[last]] + area[idx[0:last - 1]] - isec)
				idx = np.delete(idx, last, 0)
				idx = idx[o<thresh]

			pick = pick[0:counter]
			return preds[pick,:]
		
		nms_preds = np.zeros((0,6))
		for cls in range(C):
			idx = preds[:,-1] == cls
			cls_preds = preds[idx]
			cls_preds = _nms(cls_preds, overlap)
			nms_preds = np.concatenate((nms_preds, cls_preds), axis=0)
		return nms_preds

	@staticmethod
	def draw_coord(im, coords, label_map = None):
		coords = np.copy(coords)
		if im.max() <= 1:
			im = im * 255
		if im.dtype != np.uint8:
			im = im.astype(np.uint8)
		im = Image.fromarray(im)

		if coords.shape[0] == 0:
			return im

		draw = ImageDraw.Draw(im)

		unique_classes = np.unique(coords[:,-1])
		for cls in unique_classes:
			class_idx = coords[:,-1] == cls
			class_coords = coords[class_idx]
			color = tuple(np.int_(255 * np.random.rand(3,))) # color for class
			for i in range(class_coords.shape[0]):
				coord = class_coords[i,:4]
				coord[[0,2]] *= im.size[1]
				coord[[1,3]] *= im.size[0]
				coord = np.int_(coord).tolist()
				draw.rectangle(coord, outline=color)
				text = 'confidence: %.2f' % class_coords[i, -2]
				if label_map is not None:
					text = '%s, %s' % (label_map(class_coords[i,-1]), text)
				draw.text([coord[0], coord[1] - 10], text, fill=color)

		return im














