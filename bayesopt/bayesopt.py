import numpy as np
import numpy.random as npr
from scipy import linalg
from scipy.stats import norm
from scipy.optimize import brute, fmin_slsqp
from kernels import *
from samplers import *
from priors import *
from utils import format_data, sobol_seq

import pdb

# define constants
SEARCH_SOBOL = 0
SEARCH_GRID = 1

ACQ_EI = 0
ACQ_UCB = 1
ACQ_PI = 2

class BayesOpt(object):

	def __init__(
			self,
			optfun,
			X,
			Y,
			noise,
			kernel,
			bounds=None,
			burnin=500,
			resample=50,
			n_init=1,
			tol=1e-6,
			sobol_seed=1991,
			sampler=MDSliceSampler,
			sampler_args={}
		):
		assert(isinstance(kernel, BaseKernel))

		self._dim = X.shape[1] if X.shape.__len__() > 1 else 1 # get dimension of input space
		self.optfun = optfun
		self.X = np.copy(format_data(X, dim=self._dim))
		self.Y = np.copy(format_data(Y, dim=1))
		self.noise = noise
		self.kernel = kernel
		self.burnin = burnin
		self.resample = resample
		self.tol = tol
		self._sobol_seed = sobol_seed

		if bounds == None:
			bounds = []
			for i in range(self._dim):
				bounds.append((0.,1.))
		assert(len(bounds) == self._dim)
		self.bounds = bounds

		if self.X.shape[0] == 0:
			X_new, Y_new = self._random_search(n_init)
			self.X, self.Y = np.concatenate((self.X, X_new), axis=0), np.concatenate((self.Y, Y_new), axis=0)

		self._nu = npr.randn(self.X.shape[0],1)

		self._has_noise_prior = False
		if isinstance(noise, Parameter):
			self._has_noise_prior = True

		# get initial values of kernel hyperparameters
		kernel_parameters = kernel.get_valid_parameters()
		x0 = np.zeros(len(kernel_parameters) + (1 if self._has_noise_prior else 0))
		for i, par in enumerate(kernel_parameters):
			x0[i] = par.value
		if self._has_noise_prior:
			x0[-1] = self.noise.value

		self._recompute = True

		# get bounds on kernel parameters, see if you should marginalize
		bounds = []
		for par in kernel_parameters:
			bounds.append(par.prior.support)
		if self._has_noise_prior:
			bounds.append(self.noise.prior.support)

		if bounds.__len__() > 0:
			self.sampler = sampler(self._parameter_posterior, x0, bounds=bounds, log=True, burnin=burnin, **sampler_args)
			self.marginalize = True
		else:
			self.sampler = None
			self.marginalize = False

	def optimize(
			self,
			iters=20,
			verbose=False,
			num_samples=10,
			marginalize=True,
			num_grid=1000,
			search_type=SEARCH_SOBOL,
			acq_type=ACQ_EI,
			kappa=0.5,
			squash=None
		):
		bounds = self.bounds
		# determine whether or not to marginalize kernel hyperparameters
		marginalize &= self.marginalize

		if acq_type == ACQ_EI:
			acq_fun = self._ei
		elif acq_type == ACQ_UCB:
			acq_fun = lambda x: self._ucb(x, kappa)
		else:
			raise Exception('acq_type={} not valid'.format(acq_type))

		if marginalize:
			optfun = lambda x, *args: self._integrated_acquisition(x, num_samples, acq_fun)
		else:
			optfun = lambda x, *args: acq_fun

		for i in range(iters):
			if search_type == SEARCH_SOBOL:
				optval = self._sobol_search(optfun, num_grid, bounds=bounds, squash=squash)
			elif search_type == SEARCH_GRID:
				optval = self._grid_search(optfun, num_grid, bounds=bounds, squash=squash)
			else:
				raise Exception('search_type={} not valid'.format(search_type))

			self.add_observation(optval, self.optfun(optval))
			if verbose:
				print('Iteration {} complete, new point at {}, minimum at {}, value of {}'.format(i, optval, self.X[np.argmin(self.Y),:], self.Y.min()))

		max_idx = np.argmin(self.Y)
		return self.X[max_idx,:], self.Y[max_idx]

	def regress(self, x, num_samples=10, marginalize=True):
		x = format_data(x, dim=self._dim)

		if self.marginalize and marginalize:
			means = np.zeros((x.shape[0], num_samples))
			vars = np.zeros((x.shape[0], num_samples))
			hp_samples = self.sampler.sample(num_samples)
			for i in range(num_samples):
				self.set_kernel_parameters(hp_samples[i])
				mean, var = self._gp_posterior(x)

				means[:,i] = mean
				vars[:,i] = var

			mean = means.mean(axis=1)
			var = vars.mean(axis=1)
			# var = mean**2 - (means**2 + vars).mean(axis=1)
		else:
			mean, var = self._gp_posterior(x)

		return mean, var

	def set_kernel_parameters(self, values):
		self._recompute = True
		if self._has_noise_prior:
			self.kernel.set_parameters(values[:-1])
			self.noise.value = values[-1]
		else:
			self.kernel.set_parameters(values)

	def add_observation(self, X_new, Y_new):
		X_new, Y_new = format_data(X_new, dim=self._dim), format_data(Y_new, dim=1)
		if np.sqrt(np.sum((self.X - X_new)**2, axis=1)).min() > self.tol:
			self.X = np.concatenate((self.X, X_new), axis=0)
			self.Y = np.concatenate((self.Y, Y_new), axis=0)
			self._nu = npr.randn(self.X.shape[0], 1)
			if self.marginalize and self.sampler is not None:
				_ = self.sampler.sample(self.resample) # re-sample
			self._recompute = True

	def _gp_posterior(self, x):
		'''
		Using Algorithm 2.1 from Gaussian Processes for Machine Learning.

		Returns:
		mean, variance
		'''
		X, Y, noise, kernel = self.X, self.Y, self.noise, self.kernel
		if self._has_noise_prior:
			noise = noise.value

		x = format_data(x, dim=self._dim)

		if self._recompute:
			self._compute_aux()
		L, lower, alpha, ll = self._L, self._lower, self._alpha, self._ll

		K_x = kernel.compute_covariance(x, X)
		v = linalg.solve_triangular(L, K_x, lower=lower)

		mean = K_x.transpose().dot(alpha)

		var = kernel.compute_covariance(x, x, diag=True)[:,0] - (v * v).sum(axis=0) + noise # compute ONLY the diagonal

		return mean[:,0], var

	def _gp_loglikelihood(self):
		if self._recompute:
			self._compute_aux()
		return self._ll
		# return self._ll_whitened

	def _parameter_posterior(self, values):
		'''
		Compute the parameter posterior in order to marginalize the 
		kernel hyperparameters
		'''
		self.set_kernel_parameters(values)
		val = self._gp_loglikelihood() + self.kernel.logprior()
		if self._has_noise_prior:
			val += self.noise.logprob()

		return val

	def _ei(self, x):
		y_best = np.max(self.Y)
		mu, var = self._gp_posterior(x)
		gamma = (y_best - mu)/ np.sqrt(var)
		return np.sqrt(var) * (gamma * norm.cdf(gamma) + norm.pdf(gamma))

	def _ucb(self, x, kappa):
		mu, var = self._gp_posterior(x)
		return -(mu - kappa * np.sqrt(var))

	def _pi(self, x):
		y_best = np.max(self.Y)
		mu, var = self._gp_posterior(x)
		return (y_best - mu) / np.sqrt(var)

	def _integrated_acquisition(self, x, num_samples, acq_fun):
		'''
		Marginalize kernel hyperparameters
		'''
		val = 0.
		hp_samples = self.sampler.sample(num_samples)
		for i in range(num_samples):
			self.set_kernel_parameters(hp_samples[i])
			val += acq_fun(x) / num_samples
		return val

	def _sobol_search(self, optfun, num_points, bounds=None, squash=None):
		xvalues, self._sobol_seed = sobol_seq(self._dim, num_points, 50, seed=self._sobol_seed)
		if bounds is not None:
			for i, bound in enumerate(bounds):
				if isinstance(bound, tuple):
					xvalues[:,i] = bound[0] + np.diff(bound) * xvalues[:,i]
				elif isinstance(bound, list) or isinstance(bound, np.ndarray):
					bound = np.asarray(bound)
					xvalues[:,i] = bound[npr.randint(bound.size, size=num_points)]
		if squash is not None:
			xvalues = squash(xvalues, inv=False)
		yvalues = optfun(xvalues)
		max_idx = np.argmax(yvalues)
		return xvalues[[max_idx], :]

	def _grid_search(self, optfun, num_grid, bounds=None, squash=None):
		if isinstance(num_grid, list):
			assert(num_grid.__len__() == bounds.__len__())
		else:
			assert(num_grid >= 2)
			num_grid = [num_grid for i in range(self._dim)]

		xvalues = []
		for i in range(self._dim):
			xvalue = np.linspace(0, 1, num_grid[i])
			xvalue += ((xvalue[1] - xvalue[0]) * npr.rand()) # randomly shift grid
			xvalues.append(xvalue)

		if xvalues.__len__() > 1:
			xvalues = np.meshgrid(*xvalues)
			xvalues = np.concatenate([xvalue.reshape((-1,1)) for xvalue in xvalues], axis=1)
		else:
			xvalues = xvalues[0].reshape((-1,1))

		if squash is not None:
			xvalues = squash(xvalues, inv=False)

		yvalues = optfun(xvalues)
		max_idx = np.argmax(yvalues)

		return xvalues[[max_idx], :]

	def _random_search(self, n_points):
		X_new, Y_new = np.zeros((n_points, self._dim)), np.zeros((n_points,1))

		for i in range(n_points):
			x_new = np.zeros((self._dim,))
			for d, bound in enumerate(self.bounds):
				if isinstance(bound, list):
					bound = np.asarray(bound)
					x_new[d] = bound[npr.randint(bound.size)]
				else:
					x_new[d] = bound[0] + (bound[1] - bound[0]) * npr.rand()
			X_new[i,:] = x_new
			Y_new[i] = self.optfun(x_new)

		return X_new, Y_new

	def _compute_aux(self):
		X, Y, noise, kernel = self.X, self.Y, self.noise, self.kernel
		if self._has_noise_prior:
			noise = noise.value
		self._cov = kernel.compute_covariance(X, X)
		self._L, self._lower = linalg.cho_factor(self._cov + noise * np.eye(X.shape[0]), lower=True)
		self._alpha = linalg.cho_solve((self._L, self._lower), Y)
		self._ll = -0.5 * Y.transpose().dot(self._alpha) - np.log(np.diag(self._L)).sum() - float(X.shape[0])/2 * np.log(2 * np.pi) # marginal log likelihood
		self._recompute = False

		# L, lower = linalg.cho_factor(self._cov + 1e-8 * np.eye(self.X.shape[0]), lower=True)
		# f_aux = L.dot(self._nu)
		# var = self.noise.value if isinstance(self.noise, Parameter) else self.noise
		# self._ll_whitened = -((self.Y - f_aux)**2 / 2 / var).sum() - self.Y.shape[0] * np.log(var) - (self.Y.shape[0] * np.log(2 * np.pi)) / 2

		return

