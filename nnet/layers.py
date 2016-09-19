import theano
from theano import tensor as T
from theano.tensor.signal.pool import pool_2d
import numpy as np

class AbstractNNetLayer(object):
	def get_output(self):
		raise NotImplementedError('Must implement method.')
	def set_input(self, input):
		raise NotImplementedError('Must implement method.')
	def get_params(self):
		raise NotImplementedError('Must implement method.')
	def get_output_shape(self):
		raise NotImplementedError('Must implement method.')

class ConvolutionalLayer2D(AbstractNNetLayer):
	def __init__(
			self,
			input_shape,
			filter_shape,
			num_filters,
			input=None
		):
		self.input_shape = input_shape
		self.filter_shape = filter_shape
		self.num_filters = num_filters

		if input is None:
			input = T.tensor4('input')
		self.input = input

		self._initialize_params()

		self.output = self.get_output()

	def _initialize_params(self):
		W = 0.01 * np.random.randn(
			self.num_filters,
			self.input_shape[1],
			*self.filter_shape
		).astype(theano.config.floatX)
		W = theano.shared(W, name='W', borrow=True)
		b = np.zeros((self.num_filters,), dtype=theano.config.floatX)
		b = theano.shared(b, name='b', borrow=True)
		self.W = W
		self.b = b
		self.params = [W,b]

	def get_output(self):
		output = self.input
		output = T.nnet.conv2d(
			output,
			self.W
		)
		output = output + self.b.dimshuffle('x',0,'x','x')
		return output

	def set_input(self, input):
		self.input = input
		self.output = self.get_output()

	def get_params(self):
		return self.params

	def get_output_shape(self):
		output_shape = (
			self.input_shape[0],
			self.num_filters,
			self.input_shape[2] - self.filter_shape[0] + 1,
			self.input_shape[3] - self.filter_shape[1] + 1
		)
		return output_shape

class PoolLayer2D(AbstractNNetLayer):
	def __init__(self, input_shape, pool_shape, input=None):
		self.pool_shape = pool_shape
		if input is None:
			input = T.tensor4('input')
		self.input = input
		self.input_shape = input_shape
		self.output = self.get_output()

	def get_output(self):
		return pool_2d(self.input, self.pool_shape)

	def set_input(self, input):
		self.input = input
		self.output = self.get_output()

	def get_params(self):
		return []

	def get_output_shape(self):
		output_shape = (
			self.input_shape[0],
			self.input_shape[1],
			int(np.ceil(float(self.input_shape[2]) / self.pool_shape[0])),
			int(np.ceil(float(self.input_shape[3]) / self.pool_shape[1]))
		)
		return output_shape

class FCLayer(AbstractNNetLayer):
	def __init__(
			self,
			input_shape,
			num_units,
			input=None,
		):
		self.num_units = num_units
		self.input_shape = input_shape
		if input is None:
			input = theano.matrix('input')
		self.input = input

		self._initialize_params()

		self.output = self.get_output()

	def _initialize_params(self):
		W = 0.01 * np.random.randn(
			np.prod(self.input_shape[1:]),
			self.num_units
		).astype(theano.config.floatX)
		W = theano.shared(W, name='W', borrow=True)
		b = theano.shared(0., name='b', borrow=True)
		self.W = W
		self.b = b
		self.params = [W,b]

	def get_output(self):
		if self.input_shape.__len__() > 2:
			self.input = T.flatten(self.input, outdim=2)
		return T.dot(self.input, self.W) + self.b

	def get_params(self):
		return self.params

	def set_input(self, input):
		self.input = input
		self.output = self.get_output()

	def get_output_shape(self):
		output_shape = (
			self.input_shape[0],
			self.num_units
		)
		return output_shape

class NonLinLayer(AbstractNNetLayer):
	def __init__(
			self,
			nonlin,
			input_shape=None,
			input=None,
		):
		self.nonlin = nonlin
		self.input_shape = input_shape
		if input is None:
			if self.input_shape.__len__() > 2:
				input = T.tensor4('input')
			else:
				input = T.matrix('input')
		self.input = input

		self.output = self.get_output()

	def get_output(self):
		return self.nonlin(self.input)

	def get_params(self):
		return []

	def set_input(self, input):
		self.input = input
		self.output = self.get_output()

	def get_output_shape(self):
		return self.input_shape
