import theano
from theano import tensor as T
from theano.tensor.signal.pool import pool_2d
import numpy as np

class YoloObjectDetector(object):
	'''

	'''
	def __init__(self, net_definition, num_classes, input_shape, B=2, input=None):

		self.num_classes = num_classes
		self.input_shape = input_shape
		self.B = B
		if input is None:
			input = T.tensor4('input')
		self.input = input

		self.layers = []
		new_shape = input_shape
		output = self.input
		for layer_def in net_definition:
			layer = {}
			if layer_def['type'].lower() == 'conv':
				W = theano.shared(
					0.01 * np.random.randn(layer_def['num_filters'], new_shape[1], *layer_def['filter_shape']).astype(theano.config.floatX),
					borrow=True,
					name='W'
				)
				output = T.nnet.conv2d(
					output,
					W
				)
				b = theano.shared(
					np.zeros((layer_def['num_filters'],), dtype=theano.config.floatX)
				)
				output += b.dimshuffle('x',0,'x','x')
				new_shape = (
					new_shape[0],
					layer_def['num_filters'],
					new_shape[2] - layer_def['filter_shape'][0] + 1,
					new_shape[3] - layer_def['filter_shape'][1] + 1
				)
				layer.extend([W,b])
			elif layer_def['type'].lower() == 'nonlin':
				if layer_def['nonlin_type'] == 'sigmoid':
					output = T.nnet.sigmoid(output)
				elif layer_def['nonlin_type'] == 'relu':
					output = T.nnet.relu(output)
				else:
					pass
			elif layer_def['type'].lower() == 'pool':
				output = pool_2d(
					output,
					layer_def['pool_shape']
				)
				new_shape = (
					new_shape[0],
					new_shape[1],
					int(np.ceil(float(new_shape[2]) / 2)),
					int(np.ceil(float(new_shape[2]) / 2))
				)
			elif layer_def['type'].lower() == 'fc':
				new_shape = (
					new_shape[0],
					np.prod(new_shape[1:])
				)
				output = T.reshape(new_shape)
				W = theano.shared(
					0.01 * np.random.randn(new_shape[1], layer_def['num_units']).astype(theano.config.floatX),
					borrow=True,
					name='W'
				)
				b = theano.shared(
					np.float()
				)
