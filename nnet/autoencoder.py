import theano
from theano import tensor as T
from theano.tensor.signal.pool import max_pool_2d_same_size, pool_2d
import numpy as np
from itertools import tee
from collections import OrderedDict
import pickle
from tqdm import tqdm
from updates import momentum

import pdb

def unpool(input, image_shape, pool_shape):
	'''
	unpool by replacing squares with values of 
	'''

class AutoencoderLayer(object):
	'''
	Single autoencoder layer
	'''
	def __init__(
		self,
		input_size,
		n_units,
		input=None,
		tied_weights=True,
		numpy_rng=None,
		theano_rng=None
	):
		self.input_size = input_size
		self.n_units = n_units
		self.params = []
		self.params_dict = {}

		if numpy_rng is None:
			numpy_rng = np.random.RandomState(1991)

		if theano_rng is None:
			theano_rng = T.shared_randomstreams.RandomStreams(numpy_rng.randint(2**30))
		self.theano_rng = theano_rng

		if input is None:
			input = T.matrix('input')
		self.input = input

		self.W = theano.shared(
			numpy_rng.normal(
				loc=0,
				scale=.01,
				size=(input_size,n_units),
				),
			name='W',
			borrow=True
			)
		self.params_dict['W'] = self.W
		self.params.append(self.W)
		if not tied_weights:
			self.Wp = theano.shared(
				numpy_rng.normal(
					loc=0,
					scale=.01,
					size=(input_size,n_units),
					),
				name='Wp',
				borrow=True
				)
			self.params_dict['Wp'] = self.Wp
			self.params.append(self.Wp)
		else:
			self.params_dict['Wp'] = self.W.transpose()

		self.b = theano.shared(0., name='b', borrow=True)
		self.bp = theano.shared(0., name='b', borrow=True)

		self.params_dict['b'] = self.b
		self.params_dict['bp'] = self.bp

		self.params.extend([self.b, self.bp])

	def get_encoder_output(self, input, noise_level=0.0):
		'''
		Forward pass through network
		'''
		encoded = input
		if noise_level > 0:
			encoded = encoded * self.theano_rng.binomial(
				size=encoded.shape,
				n=1,
				p=1.0 - noise_level,
				dtype=theano.config.floatX
				)
		return T.nnet.sigmoid(T.dot(encoded, self.params_dict['W']) + self.params_dict['b'])

	def get_decoder_output(self, input):
		'''
		Backward pass through network.
		'''
		decoded = input
		return T.nnet.sigmoid(T.dot(decoded, self.params_dict['Wp']) + self.params_dict['bp'])

	def get_reconstructed_input(self, input, noise_level=0.0):
		return self.get_decoder_output(self.get_encoder_output(input, noise_level))

	def get_sparsity_term(self, input, noise_level=0.0, reg=1.0, sparse=1e-3):
		'''
		Regularize based on KL divergence between hidden units and some pre-determined p value
		(this is denoted by sparse).

		Params:
		-------
		input:
			input theano variable
		noise_level:
			for denoising audtoencoder
		reg:
			regularization strength
		sparse:
			sparsity term
		'''
		reg = theano.shared(reg, name='reg', borrow=True)
		sparse = theano.shared(sparse, name='sparse', borrow=True)

		hunit = self.get_encoder_output(input, noise_level)

		sparsity_term = sparse * T.log(sparse / hunit.mean(axis=0)) + (1 - sparse) * T.log((1 - sparse) / (1 - hunit.mean(axis=0)))
		sparsity_term = sparsity_term.sum()

		return reg * sparsity_term

	def get_cost_updates(self, lr=1e-3, noise_level=0.0, reg=0.0, sparse=1e-3):
		assert(noise_level >= 0 and noise_level <= 1.0)

		lr = theano.shared(lr, name='lr', borrow=True)
		z = self.get_reconstructed_input(self.input, noise_level)

		x = self.input

		cost = ((x - z)**2).sum(axis=1).mean()

		if reg > 0:
			cost += self.get_sparsity_term(self.input, reg, sparse)

		gparams = T.grad(cost, self.params)

		updates = OrderedDict()
		for param, gparam in zip(self.params, gparams):
			updates[param] = param - lr * gparam

		return cost, updates

	def train(
		self,
		data_gen,
		num_epochs,
		lr=1e-3,
		noise_level=0.0,
		reg=0.0,
		sparse=1e-3,
		verbose=True
	):
		cost, updates = self.get_cost_updates(lr, noise_level, reg, sparse)

		if verbose:
			print('Compiling training function...')

		train_fn = theano.function([self.input], cost, updates=updates)
		loss = np.zeros((num_epochs,))
		for epoch in range(num_epochs):
			data_gen, data_gen_copy = tee(data_gen, 2)
			loss_epoch = []
			for batch in data_gen:
				loss_epoch.append(train_fn(batch))
			loss[epoch] = np.mean(loss_epoch)

			if verbose:
				print('Loss for epoch %d: %.4f' % (epoch, loss[epoch]))

			data_gen = data_gen_copy

		return loss

class AutoencoderNetwork(object):
	'''
	This is for an Autoencoder network
	(i.e. can have more than one layer).
	'''
	def __init__(
		self,
		input_size,
		n_units_list,
		input=None,
		tied_weights=True,
		numpy_rng=None,
		theano_rng=None
	):
		self.input_size = input_size
		self.params =[]
		self.param_dicts = []

		if numpy_rng is None:
			numpy_rng = np.random.RandomState(1991)

		if theano_rng is None:
			theano_rng = T.shared_randomstreams.RandomStreams(numpy_rng.randint(2**30))
		self.theano_rng = theano_rng

		if input is None:
			input = T.matrix('input')
		self.input = input

		# initialize params for each
		# layer in network
		prev_layer_size = input_size
		for n_units in n_units_list:
			layer = {}
			layer['W'] = theano.shared(
					numpy_rng.normal(
						loc=0,
						scale=.01,
						size=(prev_layer_size,n_units)
						),
					name='W',
					borrow=True
				)
			if tied_weights:
				layer['Wp'] = layer['W'].transpose()
			else:
				layer['Wp'] = theano.shared(
						numpy_rng.normal(
							loc=0,
							scale=.01,
							size=(prev_layer_size,n_units)
							),
						name='Wp',
						borrow=True
					)
			layer['b'] = theano.shared(value=0., name='b', borrow=True)
			layer['bp'] = theano.shared(value=0., name='bp', borrow=True)

			self.param_dicts.append(layer)
			# add extra weight if tied weights aren't used
			if tied_weights:
				self.params.extend([layer['W'],layer['b'],layer['bp']])
			else:
				self.params.extend([layer['W'],layer['Wp'],layer['b'],layer['bp']])

			prev_layer_size = n_units

	def get_encoder_output(self, input, noise_level=0.0):
		'''
		Forward pass through network.
		'''
		encoded = input
		if noise_level > 0.0:
			encoded = encoded * self.theano_rng.binomial(
				size=encoded.shape,
				n=1,
				p=1.0 - noise_level,
				dtype=theano.config.floatX
				)
		for layer in self.param_dicts:
			encoded = T.nnet.sigmoid(T.dot(encoded, layer['W']) + layer['b'])
		return encoded

	def get_decoder_output(self, input):
		'''
		Backward pass through network.
		'''
		decoded = input
		for layer in self.param_dicts[::-1]:
			decoded = T.nnet.sigmoid(T.dot(decoded, layer['Wp']) + layer['bp'])
		return decoded

	def get_reconstructed_input(self, input, noise_level=0.0):
		'''
		Reconstruct from input
		'''
		return self.get_decoder_output(self.get_encoder_output(input, noise_level))

	def get_sparsity_term(self, input, beta=1.0, p=1e-2):
		'''
		Sparsity term based on the Kullback-Leibler divergence between the average
		activation and some chosen sparsity term p.
		'''
		beta = theano.shared(beta, name='beta', borrow=True)
		p = theano.shared(p, name='p', borrow=True)
		sparsity_term = None
		prev_layer = input
		for layer in self.param_dicts:
			hunit = T.nnet.sigmoid(T.dot(prev_layer, layer['W']) + layer['b'])
			# KL divergence between desired beroulli r variable and learned one from hidden units
			new_term = p * T.log(p / hunit.mean(axis=0)) + (1 - p) * T.log((1 - p) / (1 - hunit.mean(axis=0)))
			new_term = new_term.sum()
			if sparsity_term is None:
				sparsity_term = new_term
			else:
				sparsity_term += new_term
			prev_layer = hunit
		return beta * sparsity_term

	def get_cost_updates(self, lr=1e-3, noise_level=0.0, beta=0.0, p=1e-2):
		assert noise_level >= 0.0 and noise_level <= 1.0

		lr = theano.shared(lr, name='lr', borrow=True)
		z = self.get_reconstructed_input(self.input, noise_level)

		x = self.input

		cost = ((x - z)**2).sum(axis=1).mean()

		if beta > 0.0:
			cost += self.get_sparsity_term(self.input, beta, p)

		gparams = T.grad(cost, self.params)

		updates = OrderedDict()
		for param, gparam in zip(self.params, gparams):
			updates[param] = param - lr * gparam # gradient descent

		return cost, updates

	def train(
		self,
		data_gen,
		num_epochs,
		lr=1e-3,
		noise_level=0.0,
		beta=0.0,
		verbose=True,
	):
		cost, updates = self.get_cost_updates(lr=lr, noise_level=noise_level, beta=beta)

		if verbose:
			print('Compiling training function...')
		train_fn = theano.function([self.input],cost, updates=updates)
		loss = np.zeros((num_epochs,))
		for epoch in range(num_epochs):
			loss_epoch = []
			data_gen, data_gen_copy = tee(data_gen, 2)
			if verbose:
				print('Epoch %d' % (epoch,))
			for batch in data_gen:
				loss_epoch.append(train_fn(batch))
			loss[epoch] = np.mean(loss_epoch)
			data_gen = data_gen_copy
			if verbose:
				print('Training loss: %.4f\n' % (loss[epoch],))

		return loss

class ConvAutoencoderLayer(object):
	'''
	'''
	def __init__(
		self,
		input_shape,
		filter_shape,
		num_filters,
		pool_shape=None,
		input=None,
		numpy_rng=None,
		theano_rng=None,
		W=None,
		b=None,
		c=None,
		batch_norm=False
	):
		self.input_shape = input_shape
		self.filter_shape = filter_shape
		self.num_filters = num_filters
		self.pool_shape = pool_shape
		self.W = W
		self.b = b
		self.c = c
		self.batch_norm = True

		if input is None:
			input = T.tensor4('input')
		self.input = input

		if numpy_rng is None:
			numpy_rng = np.random.RandomState(1991)
		self.numpy_rng = numpy_rng

		if theano_rng is None:
			theano_rng = T.shared_randomstreams.RandomStreams(numpy_rng.randint(2**30))

		self.initialize_params()

		self.params = [self.W, self.b, self.c]

		if batch_norm:
			gamma_in = T.cast(theano.shared(np.ones((num_filters,)), name='gamma_in', borrow=True), theano.config.floatX)
			gamma_out = T.cast(theano.shared(1., name='gamma_out', borrow=True), theano.config.floatX)

	def initialize_params(self):
		if self.W is None:
			initial_W = np.asarray(
				self.numpy_rng.normal(
					loc=0,
					scale=0.01,
					size=((self.num_filters,self.input_shape[1]) + self.filter_shape)
					),
				dtype=theano.config.floatX
				)
			self.W = theano.shared(value=initial_W, name='W', borrow=True)
		if self.b is None:
			self.b = theano.shared(
				np.zeros(
					(self.num_filters,),
					dtype=theano.config.floatX
					),
				name='b',
				borrow=True
				)
		if self.c is None:
			self.c = theano.shared(value=0., name='c', borrow=True)

	def get_encoder_output(self, input):
		conv = T.nnet.conv2d(
			input=input,
			filters=self.W,
			filter_flip=False,
			border_mode='valid'
			)
		if self.batch_norm:
			mu = T.mean(conv, axis=0, keepdims=True)
			var = T.mean((conv - mu)**2, axis=0, keepdims=True)
			output = (conv - mu) / T.sqrt(var + 1e-5) * self.gamma_in.dimshuffle('x',0,'x','x') + self.b.dimshuffle('x',0,'x','x')
			output = T.nnet.sigmoid(output)
		else:
			output = T.nnet.sigmoid(conv + self.b.dimshuffle('x',0,'x','x'))

		if self.pool_shape is not None:
			output = max_pool_2d_same_size(output, self.pool_shape)
		return output

	def get_decoder_output(self, input):
		conv = T.nnet.conv2d(
			input=input,
			filters=self.W.dimshuffle(1,0,2,3),
			filter_flip=True,
			border_mode='full'
			)

		if self.batch_norm:
			mu = T.mean(conv, axis=0, keepdims=True)
			var = T.mean((conv-mu)**2, axsi=0, keepdims=True)
			output = (conv - mu) / T.sqrt(var + 1e-5) * self.gamma_out + self.c
			output = T.nnet.sigmoid(output)
		else:
			output = T.nnet.sigmoid(conv + self.c)

		return output

	def get_reconstructed_input(self, input):
		return self.get_decoder_output(self.get_encoder_output(input))

	def get_cost_updates(self, lr=1e-3, type='l2', reg=None):
		lr = T.cast(theano.shared(lr, name='lr', borrow=True), theano.config.floatX)

		z = self.get_reconstructed_input(self.input)
		x = self.input

		if type.lower() == 'l2':
			cost = ((x - z)**2).sum(axis=(1,2,3)).mean()
		elif type.lower() == 'l1':
			cost = (T.abs_(x - z)).sum(axis=(1,2,3)).mean()
		else:
			print('Not a cost function')

		if reg is not None:
			if reg.lower() == 'l2':
				cost += 0.5 * (T.sum(self.W**2) + T.sum(self.b**2) + T.sum(self.c**2)) # l2 regularization

		gparams = T.grad(cost, self.params)

		updates = OrderedDict()
		for param, gparam in zip(self.params, gparams):
			updates[param] = param - lr * gparam

		return cost, updates

	def train(self, data_gen, num_epochs, cost_type='l2', lr=1e-5, m=0.9, reg='l2', verbose=True):
		print('Using cost function with l2 reg, and training via momentum method.')
		cost, updates = self.get_cost_updates(lr=lr, type=cost_type, reg=reg) # change
		updates = momentum(cost, self.params, lr, m)

		if verbose:
			print('Compiling training function...')

		train_fn = theano.function([self.input], cost, updates=updates)
		loss = np.zeros((num_epochs,))
		for epoch in range(num_epochs):
			data_gen, data_gen_copy = tee(data_gen, 2)
			loss_epoch = []
			for batch in tqdm(data_gen):
				loss_epoch.append(train_fn(batch))
				if verbose:
					print('Batch loss: %.4f' % loss_epoch[loss_epoch.__len__() - 1])
			loss[epoch] = np.mean(loss_epoch)

			if verbose:
				print('Loss for epoch %d: %.4f' % (epoch, loss[epoch]))

			data_gen = data_gen_copy

		return loss

	def set_params(self, params):
		for key in params:
			self.__dict__[key].set_value(params[key], borrow=True)

	def get_params(self):
		params = {}
		params['W'] = self.W.get_value(borrow=True)
		params['b'] = self.b.get_value(borrow=True)
		params['c'] = self.c.get_value(borrow=True)
		return params

	def save_params(self, filename):
		did_save = False
		with open(filename, 'wb') as f:
			pickle.dump(self.get_params(), f)
			did_save = True
		return did_save

	def load_params(self, filename):
		with open(filename, 'rb') as f:
			params = pickle.load(f)
		self.set_params(params)

class ConvAutoencoderStack(object):
	def __init__(self, layers):
		self.layers = layers

	@staticmethod
	def unpool(input, pool_shape, input_shape, image_shape):
		'''
		Unpool by filling non-overlapping blocks with value of
		the pooled layers.

		Params:
		-------

		pool_shape:
			Shape of pooling.

		image_shape:
			Image shape of upper layer (shape that we are trying to get), this
		should be a tuple of size 2.
		'''
		new_shape = np.asarray(pool_shape) * np.asarray(input_shape)
		new_shape = T.set_subtensor(input.shape[2:], new_shape)
		new_shape = [new_shape[i] for i in range(input.ndim)]

		unpooled = T.zeros(new_shape)
		idxr = np.arange(input_shape[0] * pool_shape[0])
		idxc = np.arange(input_shape[1] * pool_shape[1])
		for idx1 in range(pool_shape[0]):
			for idx2 in range(pool_shape[1]):
				y = np.equal(np.mod(idxr - idx1, pool_shape[0]), 0).nonzero()[0]
				x = np.equal(np.mod(idxc - idx2, pool_shape[1]), 0).nonzero()[0]
				x, y = np.meshgrid(x, y)
				unpooled = T.set_subtensor(unpooled[:,:,y,x], input)
		return unpooled[:,:,:image_shape[0],:image_shape[1]]

	def get_layer_output(self, input, layer_num, pool=True):
		'''
		Get the encoded output from a certain layer starting from
		the very top

		Params:
		-------

		layer_num:
			0-index based number of layer you want output from
		'''
		encoded = input
		for layer in self.layers[:layer_num]:
			encoded = layer.get_encoder_output(encoded)
			encoded = pool_2d(encoded, layer.pool_shape)
		layer = self.layers[layer_num]
		encoded = layer.get_encoder_output(encoded)
		if pool:
			encoded = pool_2d(encoded, layer.pool_shape)
		return encoded

	def get_reconstructed_input(self, input):
		recon = self.get_layer_output(input, len(self.layers) - 1, pool=False)
		for top_layer, layer in zip(self.layers[:-1][::-1], self.layers[1:][::-1]):
			recon = layer.get_decoder_output(recon)
			image_shape = np.asarray(top_layer.input_shape[2:]) - np.asarray(top_layer.filter_shape) + 1
			recon = ConvAutoencoderStack.unpool(recon, top_layer.pool_shape, layer.input_shape[2:], image_shape)
		recon = top_layer.get_decoder_output(recon)
		return recon







