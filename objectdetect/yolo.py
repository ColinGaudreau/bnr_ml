
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
		B,
		input=None):
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
		if input is None:
			input = T.tensor4('input')
		self.input = input
		self.input_shape = input_shape

	def _get_cost(self, probs, dims, lmbda_coord=10., lmbda_noobj = .1, iou_thresh = .1):
		lmbda_coord = T.as_tensor_variable(lmbda_coord)
		lmbda_noobj = T.as_tensor_variable(lmbda_noobj)
		iou_thresh = T.as_tensor_variable(iou_thresh)
		output = network['output']
		if isinstance(output, AbstractNNetLayer):
			output = output.get_output()

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
					pred_ijk = output[:,k*5:(k+1) * 5,i,j] # single prediction for cell and box

					# get intersecion box coordinates relative to boxes
					isec_xi = T.maximum(newdims[:,0], pred_ijk[:,0])
					isec_yi = T.maximum(newdims[:,1], pred_ijk[:,1])
					isec_xf = T.minimum(newdims[:,0] + newdims[:,2], pred_ijk[:,0] + pred_ijk[:,2])
					isec_yf = T.minimum(newdims[:,1] + newdims[:,3], pred_ijk[:,1] + pred_ijk[:,3])

					isec = T.maximum((isec_xf - isec_xi) * (isec_yf - isec_yi), 0.)

					union = newdims[:,2] * newdims[:,3] + pred_ijk[:,2] * pred_ijk[:,3] - isec

					iou = isec / union

					preds_ij.append(pred_ijk.dimshuffle(0,1,'x'))
					ious.append(iou.dimshuffle(0,'x'))

				# Determine if the image intersects with the cell
				isec_xi = T.maximum(newdims[:,0], 0.)
				isec_yi = T.maximum(newdims[:,1], 0.)
				isec_xf = T.minimum(newdims[:,0] + newdims[:,2], 1. / self.S[0])
				isec_yf = T.minimum(newdims[:,1] + newdims[:,3], 1. / self.S[1])

				isec = T.maximum((isec_xf - isec_xi) * (isec_yf - isec_yi), 0.)

				union = newdims[:,2] * newdims[:,3] + pred_ijk[:,2] * pred_ijk[:,3] - isec

				iou = isec / union

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
				cost_ij_t1 += (T.sqrt(preds_ij[:,2]) - T.sqrt(newdims[:,2]))**2 + (T.sqrt(preds_ij[:,3]) - T.sqrt(newdims[:,3]))**2
				cost_ij_t1 *= lmbda_coord

				cost_ij_t1 += lmbda_noobj * (preds_ij[:,4] - ious)**2

				cost_ij_t2 = lmbda_noobj * T.sum((probs - output[:,-self.num_classes:,i,j])**2, axis=1)

				cost_ij_t1 = T.set_subtensor(cost_ij_t1[is_box_not_in_cell], 0.)
				cost_ij_t2 = T.set_subtensor(cost_ij_t2[is_not_in_cell], 0.)

				cost += cost_ij_t1 + cost_ij_t2

				dims = unscale_dims(newdims)

		cost = cost.mean()

		return cost