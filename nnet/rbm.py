import theano
from theano import tensor as T
import numpy as np
from extra import HiddenLayer, LogisticRegression
from itertools import tee
from timeit import default_timer
import pickle
import datetime
# from collections import OrderedDict

import pdb

class RBM(object):
	'''
	Restricted Boltzmann Machine
	'''
	def __init__(
		self,
		input=None,
		n_visible=784,
		n_hidden=500,
		W=None,
		hbias=None,
		vbias=None,
		numpy_rng=None,
		theano_rng=None
	):
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		
		if numpy_rng is None:
			numpy_rng = np.random.RandomState(6543)
		
		if theano_rng is None:
			theano_rng = T.shared_randomstreams.RandomStreams(numpy_rng.randint(2 ** 30))
		
		if W is None:
			initial_W = np.asarray(
				numpy_rng.uniform(
					low=-4 * np.sqrt(6. / (self.n_hidden + self.n_visible)),
					high=4 * np.sqrt(6. / (self.n_hidden + self.n_visible)),
					size=(self.n_visible,self.n_hidden)
				),
				dtype=theano.config.floatX
			)
			W = theano.shared(value=initial_W, name='W', borrow=True)
		
		if hbias is None:
			hbias = theano.shared(
				value=np.zeros(
					self.n_hidden,
					dtype=theano.config.floatX
				),
				name='hbias',
				borrow=True
			)
		
		if vbias is None:
			vbias = theano.shared(
				value=np.zeros(
					self.n_visible,
					dtype=theano.config.floatX
				),
				name='vbias',
				borrow=True
			)
		self.input = input
		if not input:
			self.input = T.matrix('input')
		
		self.W = W
		self.hbias = hbias
		self.vbias = vbias
		self.theano_rng = theano_rng
		self.params = [self.W, self.hbias, self.vbias]
	
	def propup(self, vis):
		pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
	
	def sample_h_given_v(self, v0_sample):
		pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
		h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
											 n=1, p=h1_mean,
											 dtype=theano.config.floatX)
		return [pre_sigmoid_h1, h1_mean, h1_sample]

	def propdown(self, hid):
		pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
	
	def sample_v_given_h(self, h0_sample):
		pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
		v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
											 n=1, p=v1_mean,
											 dtype=theano.config.floatX)
		return [pre_sigmoid_v1, v1_mean, v1_sample]
	
	def gibbs_hvh(self, h0_sample):
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return [pre_sigmoid_v1, v1_mean, v1_sample,
			   pre_sigmoid_h1, h1_mean, h1_sample]

	def gibbs_vhv(self, v0_sample):
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
		return [pre_sigmoid_h1, h1_mean, h1_sample,
			   pre_sigmoid_v1, v1_mean, v1_sample]

	def free_energy(self, v_sample):
		wx_b = T.dot(v_sample, self.W) + self.hbias
		vbias_term = T.dot(v_sample, self.vbias)
		hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
		return -hidden_term - vbias_term
	
	def get_cost_updates(self, lr=.1, persistent=None, k=1):
		pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
		if persistent is None:
			chain_start = ph_sample
		else:
			chain_start = persistent
			
		(
			[
				pre_sigmoid_nvs,
				nv_means,
				nv_samples,
				pre_sigmoid_nhs,
				nh_means,
				nh_samples
			],
			updates
		) = theano.scan(
			self.gibbs_hvh,
			outputs_info=[None, None, None, None, None, chain_start],
			n_steps=k,
			name='gibbs_hvh'
		)
		
		chain_end = nv_samples[-1]
		cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
		gparams = T.grad(cost, self.params, consider_constant=[chain_end])
		
		for gparam, param in zip(gparams, self.params):
			updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
		
		if persistent:
			updates[persistent] = nh_samples[-1]
			monitoring_cost = self.get_pseudo_likelihood_cost(updates)
		else:
			monitoring_cost = self.get_pseudo_likelihood_cost(updates)
		
		return monitoring_cost, updates
	
	def get_pseudo_likelihood_cost(self, updates):
		bit_i_idx = theano.shared(value=0, name='bit_i_idx')
		xi = T.round(self.input)
		fe_xi = self.free_energy(xi)
		xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
		fe_xi_flip = self.free_energy(xi_flip)
		cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
		updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
		return cost

class DBN(object):
	'''
    Deep Belief Network
    '''
	def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
				hidden_layers_sizes = [500,500], n_outs=10):
		self.sigmoid_layers = []
		self.rbm_layers = []
		self.params = []
		self.n_layers = len(hidden_layers_sizes)
		
		assert self.n_layers > 0
		
		if not theano_rng:
			theano_rng = T.shared_randomstreams.RandomStreams(numpy_rng.randint(2 ** 30))
		
		self.x = T.matrix('x')
		self.y = T.ivector('y')
		
		for i in range(self.n_layers):
			if i == 0:
				input_size = n_ins
			else:
				input_size = hidden_layers_sizes[i-1]
			
			if i == 0:
				layer_input = self.x
			else:
				layer_input = self.sigmoid_layers[-1].output
			
			sigmoid_layer = HiddenLayer(rng=numpy_rng,
									   input=layer_input,
									   n_in=input_size,
									   n_out=hidden_layers_sizes[i],
									   activation=T.nnet.sigmoid)
			
			self.sigmoid_layers.append(sigmoid_layer)
			
			self.params.extend(sigmoid_layer.params)
			
			rbm_layer = RBM(numpy_rng=numpy_rng,
						   theano_rng=theano_rng,
						   input=layer_input,
						   n_visible=input_size,
						   n_hidden=hidden_layers_sizes[i],
						   W=sigmoid_layer.W,
						   hbias=sigmoid_layer.b)
			self.rbm_layers.append(rbm_layer)
			
		self.logLayer = LogisticRegression(
			input=self.sigmoid_layers[-1].output,
			n_in=hidden_layers_sizes[-1],
			n_out=n_outs
		)
		self.params.extend(self.logLayer.params)
		
		self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
		
		self.errors = self.logLayer.errors(self.y)
			
	def pretraining_functions(self, train_set_x, batch_size, k):
		index = T.lscalar('index')
		learning_rate = T.scalar('lr')

		n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
		batch_begin = index * batch_size
		batch_end = batch_begin + batch_size

		pretrain_fns = []

		for rbm in self.rbm_layers:
			cost,updates = rbm.get_cost_updates(learning_rate,
											   persistent=None, k=k)
			fn = theano.function(
				inputs=[index, theano.In(learning_rate, value=0.1)],
				outputs=cost,
				updates=updates,
				givens={
					self.x: train_set_x[batch_begin:batch_end]
				}
			)
			pretrain_fns.append(fn)

		return pretrain_fns

	def build_finetune_functions(self, datasets, batch_size, learning_rate):
		(train_set_x, train_set_y) = datasets[0]
		(valid_set_x, valid_set_y) = datasets[1]
		(test_set_x, test_set_y) = datasets[2]
		
		learning_rate = T.cast(theano.shared(learning_rate, borrow=True), theano.config.floatX)

		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
		n_valid_batches /= batch_size
		n_test_batches = test_set_x.get_value(borrow=True).shape[0]
		n_test_batches /= batch_size

		index = T.lscalar('index')

		gparams = T.grad(self.finetune_cost, self.params)

		updates = []
		for param, gparam in zip(self.params, gparams):
			updates.append((param, param - gparam * learning_rate))

		train_fn = theano.function(
			inputs=[index],
			outputs=self.finetune_cost,
			updates=updates,
			givens={
				self.x: train_set_x[
					index * batch_size: (index+1) * batch_size
				],
				self.y: train_set_y[
					index * batch_size: (index+1) * batch_size
				]
			}
		)

		test_score_i = theano.function(
			[index],
			self.errors,
			givens={
				self.x: test_set_x[
					index * batch_size: (index+1) * batch_size
				],
				self.y: test_set_y[
					index * batch_size: (index+1) * batch_size
				]
			}
		)

		valid_score_i = theano.function(
			[index],
			self.errors,
			givens={
				self.x: test_set_x[
					index * batch_size: (index+1) * batch_size
				],
				self.y: test_set_y[
					index * batch_size: (index+1) * batch_size
				]
			}
		)

		def valid_score():
			return [valid_score_i(i) for i in range(n_valid_batches)]
		def test_score():
			return [test_score_i(i) for i in range(n_test_batches)]

		return train_fn, valid_score, test_score

class CRBM(object):
	'''
	Convolutional Restricted Boltzmann Machine
	'''
	def __init__(
			self,
			input=None,
			input_shape=(None,1,28,28),
			filter_shape=(None,3,3,3),
			pool_shape=(2,2),
			num_filters=32,
			W=None,
			b=None,
			c=None,
			numpy_rng=None,
			theano_rng=None
	):	
		# need odd shaped filters
		assert filter_shape[0] % 2 != 0 and filter_shape[1] % 2 != 0

		if input is None:
			input = T.tensor4('input')
		self.input = input
		self.input_shape = input_shape
		self.filter_shape = filter_shape
		self.pool_shape = pool_shape
		self.num_filters = num_filters

		if numpy_rng is None:
			numpy_rng = np.random.RandomState(5824)
		self.numpy_rng = numpy_rng

		if theano_rng is None:
			theano_rng = T.shared_randomstreams.RandomStreams(numpy_rng.randint(2**30))
		self.theano_rng = theano_rng

		self.initialize_params(W,b,c)

	def initialize_params(self, W=None, b=None, c=None):
		if W is None:
			initial_W = np.asarray(
				self.numpy_rng.normal(
					loc=0,
					scale=.01,
					size=((self.num_filters,self.input_shape[1]) + self.filter_shape),
					),
				dtype=theano.config.floatX
				)
			self.W = theano.shared(value=initial_W, name='W', borrow=True)

		if c is None:
			c = theano.shared(value=0., name='c', borrow=True)
		self.c = c

		if b is None:
			b = theano.shared(
				np.zeros(
					(self.num_filters,),
					dtype=theano.config.floatX
					),
				name='b',
				borrow=True
				)
		self.b = b

		self.params = [self.W, self.b, self.c]

	def propup(self, vis):
		# first assume grayscale image (just to begin, will change soon).
		pre_sigmoid_activation = T.nnet.conv2d(
				input=vis,
				filters=self.W, # flip matrix when propogating upwards
				input_shape=self.input_shape,
				filter_flip=False,
				border_mode='half'
			)
		pre_sigmoid_activation += self.b.dimshuffle('x',0,'x','x')
		return pre_sigmoid_activation

	def sample_h_given_v(self, v0_sample, pool=True, scan=False, optimize=True):
		pre_sigmoid_h1 = self.propup(v0_sample)
		if pool:
			if scan:
				ridx_np, cidx_np = np.meshgrid(range(0,self.input_shape[2], self.pool_shape[0]),
					range(0,self.input_shape[3], self.pool_shape[1]))
				ridx_np, cidx_np = ridx_np.reshape((-1,)), cidx_np.reshape((-1,))
				ridx, cidx = theano.shared(ridx_np, name='ridx'), theano.shared(cidx_np, name='cidx')
				input_shape = theano.shared(np.asarray(self.input_shape[2:]), name='input_shape')
				pool_shape = theano.shared(np.asarray(self.pool_shape), name='pool_shape')

				def scan_fun(row, col, tensor, pre_sigmoid_h1, input_shape, pool_shape):
					inttype = 'int32'
					rrange = (T.cast(row, inttype), T.cast(T.min(T.stack(row + pool_shape[0], input_shape[0])), inttype))
					crange = (T.cast(col, inttype), T.cast(T.min(T.stack(col + pool_shape[1], input_shape[1])), inttype))
					pool_overflow = T.prod(pool_shape) - (rrange[1] - rrange[0]) * (crange[1] - crange[0])
					p = T.exp(pre_sigmoid_h1[:,:,rrange[0]:rrange[1],crange[0]:crange[1]]) / \
						(1 + pool_overflow +  T.sum(T.exp(pre_sigmoid_h1[:,:,rrange[0]:rrange[1],crange[0]:crange[1]]), axis=[3,2], keepdims=True))

					return T.set_subtensor(tensor[:,:,rrange[0]:rrange[1],crange[0]:crange[1]], p)

				output,_ = theano.scan(
					scan_fun,
					outputs_info=[T.zeros_like(pre_sigmoid_h1)],
					sequences=[ridx,cidx],
					non_sequences=[pre_sigmoid_h1,input_shape,pool_shape],
					n_steps=ridx_np.size
					)

				h1_mean = output[-1]
			elif optimize:
				get_proper_size = lambda n,m: n if n%m==0 else n + (m - n%m)
				rmax = get_proper_size(self.input_shape[2], self.pool_shape[0])
				cmax = get_proper_size(self.input_shape[3], self.pool_shape[1])
				new_pre_sigmoid = T.zeros(
					(
						pre_sigmoid_h1.shape[0],
						pre_sigmoid_h1.shape[1],
						rmax,
						cmax
					),
					dtype=theano.config.floatX
				)
				new_pre_sigmoid = T.set_subtensor(new_pre_sigmoid[:,:,:pre_sigmoid_h1.shape[2],:pre_sigmoid_h1.shape[3]], pre_sigmoid_h1)
				h1_mean = T.zeros_like(new_pre_sigmoid, dtype=theano.config.floatX)
				activation_list = []
				idx1 = np.arange(rmax)
				idx2 = np.arange(cmax)
				for ridx in range(self.pool_shape[0]):
					for cidx in range(self.pool_shape[1]):
						new_ridx, new_cidx = idx1[np.equal(np.mod(idx1 - ridx, self.pool_shape[0]),0)], idx2[np.equal(np.mod(idx2 - cidx, self.pool_shape[1]),0)]
						new_cidx, new_ridx = np.meshgrid(new_ridx, new_cidx)
						activation_list.append(T.exp(new_pre_sigmoid[:,:,new_ridx,new_cidx]))
				activation_denom = sum(activation_list) + 1
				cnt = 0
				for ridx in range(self.pool_shape[0]):
					for cidx in range(self.pool_shape[1]):
						new_ridx, new_cidx = idx1[np.equal(np.mod(idx1 - ridx, self.pool_shape[0]),0)], idx2[np.equal(np.mod(idx2 - cidx, self.pool_shape[1]),0)]
						new_cidx, new_ridx = np.meshgrid(new_ridx, new_cidx)
						h1_mean = T.set_subtensor(h1_mean[:,:,new_ridx,new_cidx], activation_list[cnt] / activation_denom)
						cnt += 1
				h1_mean = h1_mean[:,:,:self.input_shape[2],:self.input_shape[3]]
			else:
				h1_mean = T.zeros_like(pre_sigmoid_h1)
				for ridx in range(0, self.input_shape[2], self.pool_shape[0]):
					for cidx in range(0, self.input_shape[3], self.pool_shape[1]):
						rrange = (ridx, min(ridx + self.pool_shape[0], self.input_shape[2]))
						crange = (cidx, min(cidx + self.pool_shape[1], self.input_shape[3]))
						pool_overflow = np.prod(self.pool_shape) - (rrange[1] - rrange[0]) * (crange[1] - crange[0])
						p = T.exp(pre_sigmoid_h1[:,:,rrange[0]:rrange[1],crange[0]:crange[1]]) / \
							(1 + pool_overflow +  T.sum(T.exp(pre_sigmoid_h1[:,:,rrange[0]:rrange[1],crange[0]:crange[1]]), axis=[3,2], keepdims=True))
						h1_mean = T.set_subtensor(h1_mean[:,:,rrange[0]:rrange[1],crange[0]:crange[1]], p)
		else:
			h1_mean = T.nnet.sigmoid(pre_sigmoid_h1)

		h1_sample = self.theano_rng.binomial(
			size=h1_mean.shape,
			n=1,
			p=h1_mean,
			dtype=theano.config.floatX
			)

		return [pre_sigmoid_h1, h1_mean, h1_sample]

	def sample_p_given_v(self, v0_sample, scan=False, optimize=True):
		pre_sigmoid_h1 = self.propup(v0_sample)

		p1shape = (
				pre_sigmoid_h1.shape[0],
				pre_sigmoid_h1.shape[1],
				np.int_(np.ceil(np.float_(self.input_shape[2]) / self.pool_shape[0])),
				np.int_(np.ceil(np.float_(self.input_shape[3]) / self.pool_shape[1]))
			)

		if scan:
			ridx_np, cidx_np = range(0, self.input_shape[2], self.pool_shape[0]), range(0, self.input_shape[3], self.pool_shape[1])
			i_np, j_np = range(len(ridx_np)), range(len(cidx_np))
			i_np,j_np = np.meshgrid(i_np, j_np)
			i_np,j_np = i_np.reshape((-1,)), j_np.reshape((-1,))
			ridx_np,cidx_np = np.meshgrid(ridx_np, cidx_np)
			ridx_np,cidx_np = ridx_np.reshape((-1,)), cidx_np.reshape((-1,))

			i = theano.shared(i_np, name='i')
			j = theano.shared(j_np, name='j')
			ridx = theano.shared(ridx_np, name='ridx')
			cidx = theano.shared(cidx_np, name='cidx')

			input_shape = theano.shared(np.asarray(self.input_shape[2:]), name='input_shape')
			pool_shape = theano.shared(np.asarray(self.pool_shape), name='pool_shape')

			def scan_fun(i,j,row,col,tensor,pre_sigmoid_h1,input_shape,pool_shape):
				inttype = 'int32'
				rrange = (T.cast(row, inttype), T.cast(T.min(T.stack(row + pool_shape[0], input_shape[0])), inttype))
				crange = (T.cast(col, inttype), T.cast(T.min(T.stack(col + pool_shape[1], input_shape[1])), inttype))
				pool_overflow = T.prod(pool_shape) - (rrange[1] - rrange[0]) * (crange[1] - crange[0])

				return T.set_subtensor(tensor[:,:,i,j], pool_overflow + T.sum(T.exp(pre_sigmoid_h1[:,:,rrange[0]:rrange[1],crange[0]:crange[1]]), axis=[3,2]))

			output,_ = theano.scan(
				scan_fun,
				outputs_info=[T.zeros(p1shape)],
				sequences=[i,j,ridx,cidx],
				non_sequences=[pre_sigmoid_h1,input_shape,pool_shape],
				n_steps=i_np.size
				)

			p1_mean = output[-1]
		elif optimize:
			get_proper_size = lambda n,m: n if n%m==0 else n + (m - n%m)
			rmax = get_proper_size(self.input_shape[2], self.pool_shape[0])
			cmax = get_proper_size(self.input_shape[3], self.pool_shape[1])
			new_pre_sigmoid = T.zeros(
				(
					pre_sigmoid_h1.shape[0],
					pre_sigmoid_h1.shape[1],
					rmax,
					cmax
				),
				dtype=theano.config.floatX
			)
			new_pre_sigmoid = T.set_subtensor(new_pre_sigmoid[:,:,:pre_sigmoid_h1.shape[2],:pre_sigmoid_h1.shape[3]], pre_sigmoid_h1)
			activation_list = []
			idx1 = np.arange(rmax)
			idx2 = np.arange(cmax)
			for ridx in range(self.pool_shape[0]):
				for cidx in range(self.pool_shape[1]):
					new_ridx, new_cidx = idx1[np.equal(np.mod(idx1 - ridx, self.pool_shape[0]),0)], idx2[np.equal(np.mod(idx2 - cidx, self.pool_shape[1]),0)]
					new_cidx, new_ridx = np.meshgrid(new_ridx, new_cidx)
					activation_list.append(T.exp(new_pre_sigmoid[:,:,new_ridx,new_cidx]))
			p1_mean = sum(activation_list) / (1 + sum(activation_list))
		else:
			p1_mean = T.alloc(0., *p1shape)

			for i, ridx in enumerate(range(0, self.input_shape[2], self.pool_shape[0])):
				for j, cidx in enumerate(range(0, self.input_shape[3], self.pool_shape[1])):
					rrange = (ridx, min(ridx + self.pool_shape[0], self.input_shape[2]))
					crange = (cidx, min(cidx + self.pool_shape[1], self.input_shape[3]))
					pool_overflow = np.prod(self.pool_shape) - (rrange[1] - rrange[0]) * (crange[1] - crange[0])
					p1_mean = T.set_subtensor(p1_mean[:,:,i,j], pool_overflow + T.sum(T.exp(pre_sigmoid_h1[:,:,rrange[0]:rrange[1],crange[0]:crange[1]]), axis=[3,2]))
			p1_mean = T.cast(p1_mean, theano.config.floatX) # not sure why I need to cast :(

		p1_mean = p1_mean / (1 + p1_mean)
		p1_sample = self.theano_rng.binomial(
				size=p1_mean.shape,
				n=1,
				p=p1_mean,
				dtype=theano.config.floatX
			)

		return [p1_mean, p1_sample]

	def propdown(self, hid):
		pre_sigmoid_activation = T.nnet.conv2d(
				input=hid,
				filters=self.W.dimshuffle(1,0,2,3),
				input_shape=self.input_shape,
				border_mode='half',
			)
		pre_sigmoid_activation += self.c
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

	def sample_v_given_h(self, h0_sample):
		pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
		v1_sample = self.theano_rng.binomial(
				size=v1_mean.shape,
				n=1,
				p=v1_mean,
				dtype=theano.config.floatX
			)
		return [pre_sigmoid_v1, v1_mean, v1_sample]

	def gibbs_hvh(self, h0_sample, scan=False, optimize=True):
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample, scan, optimize)
		return [pre_sigmoid_v1, v1_mean, v1_sample,
			   pre_sigmoid_h1, h1_mean, h1_sample]

	def gibbs_vhv(self, v0_sample, scan=False, optimize=True):
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample, scan, optimize)
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
		return [pre_sigmoid_h1, h1_mean, h1_sample,
			   pre_sigmoid_v1, v1_mean, v1_sample]

	def free_energy(self, v_sample):
		wx_b = T.nnet.conv2d(
				input=v_sample,
				filters=self.W,
				filter_shape=self.filter_shape,
				input_shape=self.input_shape,
				filter_flip=False,
				border_mode='half'
			)
		wx_b = wx_b + self.b.dimshuffle('x',0,'x','x')
		hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=[3,2,1])
		vbias_term = self.c * T.sum(v_sample, axis=[3,2,1])
		return -hidden_term - vbias_term

	def get_cost_updates(self, momentum=.9, lr=1e-3, k=1, reg_fact=1.0, sparse_fact=1e-4, cost_type='reconstruction', scan=False):
		'''
		Gets the cost function for monitoring performance as well as the update parameters 
		Monitoring cost function is different from the cost function that is optimized. In this
		case we use the entropy caused by switching a bit in the input.

		Parameters
		----------
		momentum:
			Momentum factor for momentum based gradient descent
		lr:
			Learning rate
		k:
			Number of samples to take when doing block Gibbs sampling
		reg_fact:
			Regularization factor
		sparse_fact:
			Sparsity factor; greater values correspond to sparser filters.
		'''
		pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input, scan=scan)
		chain_start = ph_sample

		(
			[
				pre_sigmoid_nvs,
				nv_means,
				nv_samples,
				pre_sigmoid_nhs,
				nh_means,
				nh_samples
			],
			updates
		) = theano.scan(
			lambda input: self.gibbs_hvh(input, scan),
			outputs_info=[None,None,None,None,None,chain_start],
			n_steps=k,
			name='gibbs_hvh'
		)

		chain_end = nv_samples[-1]
		cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
		cost += self.get_sparsity_term(reg_fact, sparse_fact)

		gparams = T.grad(cost, self.params, consider_constant=[chain_end])

		momentum = T.cast(momentum, dtype=theano.config.floatX)
		lr = T.cast(lr, dtype=theano.config.floatX)

		for gparam, param in zip(gparams, self.params):
			value = param.get_value(borrow=True)
			velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
				broadcastable=param.broadcastable)
			updates[velocity] = momentum * velocity - lr * gparam
			updates[param] = param + momentum * velocity - lr * gparam

		if cost_type == 'reconstruction':
			cost = self.get_reconstruction_cost()
		elif cost_type == 'plikelihood':
			cost = self.get_pseudo_likelihood_cost(updates)
		else:
			raise Exception('cost_type "{}" is invalid.'.format(cost_type))

		return cost, updates

	def get_sparsity_term(self, reg_fact, sparse_fact):
		'''
		Get sparsity term, choose eighter l1 or l2 norm. Sparsity term 
		is defined as the average activation across all hidden units.
		'''
		_, hmean, _ = self.sample_h_given_v(self.input)
		sparsity_term = reg_fact * ((sparse_fact - hmean.mean(axis=0))**2).sum()
		return sparsity_term

	def get_pseudo_likelihood_cost(self, updates):
		bit_i_idx = theano.shared(value=0, name='bit_i_idx')
		xi = T.round(self.input)
		fe_xi = self.free_energy(xi)

		row_size = self.input_shape[2]
		col = T.floor(bit_i_idx // row_size)
		row = bit_i_idx % row_size

		xi_flip = T.set_subtensor(xi[:,:,row,col], 1 - xi[:,:,row,col])
		fe_xi_flip = self.free_energy(xi_flip)
		cost = T.mean(np.prod(self.input_shape[2:]) * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
		updates[bit_i_idx] = (bit_i_idx + 1) % np.prod(self.input_shape[2:])
		return cost

	def get_reconstruction_cost(self):
		'''
		Compute average reconstruction error of batch
		'''
		_,_,_,_,newvis,_ = self.gibbs_vhv(self.input)
		return T.mean(T.sum((newvis - self.input)**2, axis=[3,2,1]))

class CDBN(object):
	'''
	Convolutional Deep Belief Network
	'''
	def __init__(
		self,
		input=None,
		input_shape=(None,1,28,28),
		hidden_layers_num_filters=[32],
		hidden_layers_filter_shapes=[(3,3)],
		hidden_layers_pool_shapes=[(2,2)],
		params=None
	):
		'''
		Initialize each individual CRBM, calculating the shape of the inputs for the
		hidden layers.

		Parameters
		----------
		params:
			List of dictionaries with pre-computed params
		'''
		assert len(hidden_layers_num_filters) == len(hidden_layers_filter_shapes)
		assert len(hidden_layers_num_filters) == len(hidden_layers_pool_shapes)

		self.input_shape = input_shape
		self.hidden_layers_num_filters = hidden_layers_num_filters
		self.hidden_layers_filter_shapes = hidden_layers_filter_shapes
		self.hidden_layers_pool_shapes = hidden_layers_pool_shapes

		self.n_layers = len(self.hidden_layers_num_filters)
		self.crbm_layers = []

		if not input:
			input = T.tensor4('input')
		self.input = input

		for idx in range(self.n_layers):
			if idx == 0:
				input_shape = self.input_shape
			else:
				prev_layer_input_shape = self.crbm_layers[idx-1].input_shape
				prev_layer_pool_shape = self.hidden_layers_pool_shapes[idx-1]
				# calculate shape of new layer
				input_shape = (
						None,
						self.hidden_layers_num_filters[idx-1],
						np.int_(np.ceil(np.float_(prev_layer_input_shape[2]) / prev_layer_pool_shape[0])),
						np.int_(np.ceil(np.float_(prev_layer_input_shape[3]) / prev_layer_pool_shape[1])),
					)

			new_crbm = CRBM(
				input_shape=input_shape,
				filter_shape=self.hidden_layers_filter_shapes[idx],
				pool_shape=self.hidden_layers_pool_shapes[idx],
				num_filters=self.hidden_layers_num_filters[idx]
				)

			self.crbm_layers.append(new_crbm)

		if params is not None:
				self.set_params(params)

	def train(
		self,
		data_gen,
		num_epochs,
		train_args=None,
		representative_data=None,
		use_mean=True,
		verbose=False,
		save_every=1,
		weights_file=None,
		loss_file=None,
		layers_to_train=None,
	):
		'''
		Train layers greedily.

		Parameters
		----------
		data_gen:
			Generator over batches (this way you don't necessarily load all data into memory at once,
			which can fuck everything up when running on the GPU).

		train_args:
			List of dictionaries with arguments for training (i.e. momentum, learning rate, etc.).

		representative_data:
			Two subsets of the data; one from the training set, the other from the validation set.  We use these to monitor overfitting
			by looking that the average free energy of these two sets (see Hinton's guid to training RBMs).

		save_every:
			Save the parameters every save_every epochs.

		weights_file:
			Name of file for saving parameters.

		layers_to_train:
			List of layers to train, if None then train all the layers.


		'''
		assert len(num_epochs) == len(self.crbm_layers)
		assert train_args is None or len(train_args) == len(self.crbm_layers)

		if layers_to_train is None:
			layers_to_train = range(self.num_layers)

		if train_args is None:
			train_args = [{} for num_layers in range(len(self.crbm_layers))]

		losses = [0] * len(self.crbm_layers)
		for layer_num in range(len(self.crbm_layers)):
			crbm = self.crbm_layers[layer_num]
			if layer_num in layers_to_train:
				loss = np.zeros((num_epochs[layer_num],))
				cost, updates = crbm.get_cost_updates(**train_args[layer_num])
				train_fn = theano.function(
					[crbm.input],
					cost,
					updates=updates,
					name='train_fn'
					)

				if representative_data is not None:
					input1 = T.tensor4('rep_train_set')
					input2 = T.tensor4('rep_test_set')
					avg_fe_div = T.mean(abs(crbm.free_energy(input1) - crbm.free_energy(input2)))
					avg_fe_div = theano.function([], avg_fe_div, givens={input1: representative_data[0], input2: representative_data[1]})

				if verbose:
					print('\nTraining layer {}\n'.format(layer_num))
				
				for epoch in range(num_epochs[layer_num]):
					data_gen, data_gen_copy = tee(data_gen,2)
					tmp = []
					if verbose:
						ti = default_timer()
					for idx,batch in enumerate(data_gen):
						tmp.append(train_fn(batch))
					loss[epoch] = np.mean(tmp)
					data_gen = data_gen_copy

					losses[layer_num] = loss

					if verbose:
						print('Layer {}, Epoch {}, Loss: {}'.format(layer_num, epoch, loss[epoch]))
						if representative_data is not None:
							print ('Average Free Energy Difference Between Training and Test Set: {}.'.format(avg_fe_div()))
						print('\n Training took %.2f seconds.' % (default_timer() - ti,))
						print(datetime.datetime.now())

					if epoch % save_every == 0 and weights_file is not None:
						self.save_params(weights_file)
					if epoch % save_every == 0 and loss_file is not None:
						with open(loss_file, 'wb') as f:
							pickle.dump(losses, f)

				losses.append(loss)

			# define new gen to propagate through net
			batch_input = T.tensor4('batch_input')
			vsample,_ = crbm.sample_p_given_v(batch_input)
			vsample = vsample / T.max(vsample)
			batch_function = theano.function([batch_input], vsample)
			def new_gen(data_gen):
				for batch in data_gen:
					yield batch_function(batch).astype(theano.config.floatX)
			data_gen, data_gen_copy = tee(new_gen(data_gen), 2)

			# propagate rep. data to the next layer
			if representative_data is not None:
				representative_data[0] = batch_function(representative_data[0])
				representative_data[1] = batch_function(representative_data[1])

		return losses

	def load_params(self, filename):
		with open(filename, 'rb') as f:
			weights = pickle.load(f)
		self.set_params(weights)

	def save_params(self, filename):
		did_save = False
		with open(filename, 'wb') as f:
			pickle.dump(self.get_params(), f)
			did_save = True
		return did_save

	def get_params(self):
		params = []
		for crbm in self.crbm_layers:
			layer = {}
			for param in crbm.params:
				key = param.__str__()
				layer[key] = param.get_value(borrow=True)
			params.append(layer)
		return params

	def set_params(self, params):
		for idx in range(len(params)):
			crbm = self.crbm_layers[idx]
			for key in params[idx]:
				param = params[idx][key]
				if not np.isscalar(param):
					param = param.astype(theano.config.floatX)
				crbm.__dict__[key].set_value(param, borrow=True)

