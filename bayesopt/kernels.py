import numpy as np
from scipy.special import kv, gamma
from scipy.stats import beta
from utils import format_data
from priors import *
import copy
import bayesopt

import pdb

class Parameter(object):
	def __init__(self, value, prior=None):
		self.value = value
		self.prior = prior

	def logprob(self):
		return self.prior.logprob(self.value)

	def __repr__(self):
		return 'Parameter({}, {})'.format(self.value, self.prior)

class BaseKernel(object):
	def __init__(self, parameters=[]):
		self.parameters = parameters

	def compute_covariance(self, x, y):
		raise NotImplementedError

	def logprior(self):
		val = 0.
		for par in self.get_valid_parameters():
			val += par.prior.logprob(par.value)
		return val

	def get_parameters(self):
		return np.asarray([par.value for par in self.parameters])

	def set_parameters(self, values):
		valid_parameters = self.get_valid_parameters()
		assert(len(valid_parameters) == values.size)
		for i, par in enumerate(valid_parameters):
			par.value = values[i]

	def get_valid_parameters(self):
		return [par for par in self.parameters if par.prior is not None]

	def _format_vects(self, x, y, diag=False):
		x, y = format_data(x), format_data(y)

		if not diag:
			idx1, idx2 = np.meshgrid(np.arange(x.shape[0]), np.arange(y.shape[0]))
			x, y =  x[idx1], y[idx2]
		else:
			x, y = x.reshape(x.shape + (1,)).swapaxes(1,2), x.reshape(x.shape + (1,)).swapaxes(1,2)

		return x, y

	def __repr__(self):
		return 'Kernel({}, {})'.format(self.__class__, self.parameters.__str__())

class SquaredExpKernel(BaseKernel):
	def __init__(self, parameters=[]):
		if parameters.__len__() == 0:
			self.parameters = [Parameter(0.1, GammaPrior())]
		else:
			for par in parameters:
				assert(isinstance(par, Parameter))
			self.parameters = parameters

	def compute_covariance(self, x, y, diag=False):
		x, y = self._format_vects(x, y, diag=diag)
		cov = 0
		if self.parameters.__len__() < 2:
			cov = np.exp(-((x - y)**2).sum(axis=2) / self.parameters[0].value)
		else:
			cov = self.parameters[1].value * np.exp(-((x - y)**2).sum(axis=2) / self.parameters[0].value)

		if self.parameters.__len__() > 2:
			cov += (self.parameters[2].value * np.eye(cov.shape[0]))

		return cov

class AbsExpKernel(BaseKernel):
	def __init__(self, parameters=[]):
		if parameters.__len__() == 0:
			self.parameters = [Parameter(1., GammaPrior())]
		else:
			for par in parameters:
				assert(isinstance(par, Parameter))
			self.parameters = parameters

	def compute_covariance(self, x, y, diag=False):
		x, y = self._format_vects(x, y, diag=diag)
		cov = 0
		if self.parameters.__len__() < 2:
			cov = np.exp(-np.abs((x - y)).sum(axis=2) / self.parameters[0].value)
		else:
			cov = self.parameters[1].value * np.exp(-np.abs((x - y)).sum(axis=2) / self.parameters[0].value)

		if self.parameters.__len__() > 2:
			cov += (self.parameters[2].value * np.eye(cov.shape[0]))

		return cov

class LinearKernel(BaseKernel):
	def __init__(self):
		super(LinearKernel, self).__init__()

	def compute_covariance(self, x, y, diag=False):
		x, y = self._format_vects(x, y, diag=diag)
		return (x * y).sum(axis=2)


class MaternKernel(BaseKernel):
	def __init__(self, parameters=[], type='5/2'):
		if parameters.__len__() == 0:
			self.parameters = [Parameter(0.1, GammaPrior())]
		else:
			for par in parameters:
				assert(isinstance(par, Parameter))
			self.parameters = parameters
		assert(type == '5/2' or type == '3/2')
		self.type = type

	def compute_covariance(self, x, y, diag=False):
		x, y = self._format_vects(x, y, diag=diag)
		d = np.sqrt(((x - y)**2).sum(axis=2))
		l = self.parameters[0].value
		if self.type == '3/2':
			val = (1 + np.sqrt(3)*d/l) * np.exp(-np.sqrt(3)*d/l)
		else:
			val = (1 + np.sqrt(5)*d/l + (5 * d**2) / (3 * l**2)) * np.exp(-np.sqrt(5)*d/l)

		return val

class InputWarpedKernel(BaseKernel):
	'''
	Input warped kernel
	'''
	def __init__(self, kernel, N, parameters=None):
		self.kernel = kernel
		self.N = N

		if parameters is not None:
			assert(parameters.__len__() == 2 * N)
			for par in parameters:
				assert(isinstance(par, Parameter))
		else:
			parameters = []
			for i in range(N):
				parameters.extend([Parameter(1., LogNormalPrior(0., 0.5)), Parameter(1., LogNormalPrior(0., 0.5))])

		# add warping parameters
		self.parameters = []
		self.parameters.extend(kernel.parameters)
		self.parameters.extend(parameters)

	def compute_covariance(self, x, y, diag=False):
		x, y = format_data(x), format_data(y)
		x, y = np.copy(x), np.copy(y)
		assert(x.shape[1] == self.N and y.shape[1] == self.N)

		for i in range(x.shape[1]):
			a, b = self.parameters[-2*self.N + 2*i].value, self.parameters[-2*self.N + 2*i + 1].value
			x[:,i] = beta.cdf(x[:,i], a, b)
			y[:,i] = beta.cdf(y[:,i], a, b)

		val =  self.kernel.compute_covariance(x, y, diag=diag)

		return val

class DiscreteKernel(BaseKernel):
	'''
	Kernel for discrete data:
	K(x, y) = {1 if x = j, 0 otherwise}
	'''
	def __init__(self, parameters=[]):
		self.parameters = parameters
		# if parameters.__len__() == 0:
		# 	self.parameters = [Parameter(1., GammaPrior(3., 1.))]
		# else:
		# 	for par in parameters:
		# 		assert(isinstance(par, Parameter))
		# 	self.parameters = parameters

	def compute_covariance(self, x, y, diag=False):
		x, y = self._format_vects(x, y, diag=diag)

		equal_idx = np.sum(x == y, axis=2).astype(np.float64) / x.shape[2]
		# equal_idx = (np.sum(x == y, axis=2) == x.shape[2]).astype(np.float64)

		return equal_idx

class MixedKernel(BaseKernel):
	'''
	Use a different kernel for different sorts of kernels.
	'''
	def __init__(self, feature_map):
		'''
		feature_map: list of tuples where the first entry is a kernel, and the second is the corresponding variable indices as an numpy.ndarray.
		'''
		super(MixedKernel, self).__init__()
		self.kernels, self.maps, self.parameters = [], [], []
		for kernel, mp in feature_map:
			self.kernels.append(kernel)
			self.maps.append(mp)
			self.parameters.extend(kernel.parameters)

	def compute_covariance(self, x, y, diag=False):
		x, y = format_data(x), format_data(y)

		covs = []
		for kernel, idx in zip(self.kernels, self.maps):
			covs.append(kernel.compute_covariance(x[:,idx], y[:,idx], diag=diag))

		return np.prod(covs, axis=0)






