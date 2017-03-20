import theano
from theano import tensor as T
import lasagne.layers as layers
import numpy as np

import pdb

class Upsampling2DLayer(layers.Layer):

	def __init__(self, incoming, output_shape, **kwargs):
		super(Upsampling2DLayer, self).__init__(incoming, **kwargs)
		assert(output_shape[0] >= self.input_shape[0] and output_shape[1] >= self.input_shape[1])
		assert(self.input_shape.__len__() == 4)
		self._output_shape = output_shape

	def get_output_shape_for(self, input_shape):
		return input_shape[:2] + self._output_shape

	def get_output_for(self, input, **kwargs):
		ratio = int(np.ceil(float(self._output_shape[0]) / self.input_shape[-2]))
		print(ratio)
		output = T.nnet.abstract_conv.bilinear_upsampling(input, ratio)
		return output[:,:,:self._output_shape[0],:self._output_shape[1]]