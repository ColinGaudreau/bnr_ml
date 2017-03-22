import theano
from theano import tensor as T
import lasagne.layers as layers
import numpy as np
from bnr_ml.utils.theano_extensions import bilinear_upsampling

import pdb

class Upsampling2DLayer(layers.Layer):

	def __init__(self, incoming, output_shape, **kwargs):
		super(Upsampling2DLayer, self).__init__(incoming, **kwargs)
		assert(output_shape[0] >= self.input_shape[-2] and output_shape[1] >= self.input_shape[-1])
		assert(self.input_shape.__len__() == 4)
		self._output_shape = output_shape

	def get_output_shape_for(self, input_shape):
		return input_shape[:2] + self._output_shape

	def get_output_for(self, input, **kwargs):
		ratio = int(np.ceil(float(self._output_shape[0]) / self.input_shape[-2]))
		output = bilinear_upsampling(input, ratio, use_1D_kernel=False)
		return output[:,:,:self._output_shape[0],:self._output_shape[1]]
