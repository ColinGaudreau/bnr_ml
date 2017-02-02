import numpy as np
from scipy import linalg
from scipy.stats import norm
from scipy.optimize import brute, fmin_slsqp
from kernels import *
from samplers import *
from priors import *
from utils import format_data

import pdb

class BayesOpt(object):

	def __init__(self, optfun, X, Y, noise, kernel):
		assert(isinstance(kernel, BaseKernel))

		self.optfun = optfun
		self.X = np.copy(format_data(X))
		self.Y = np.copy(format_data(Y))
		self.noise = noise
		self.kernel = kernel

		x0 = np.zeros(kernel.parameters.__len__())
		for i in range(x0.size):
			x0[i] = kernel.parameters[i].value

		self._recompute = True

		# get bounds on kernel parameters, see if you should marginalize
		bounds = []
		for par in self.kernel.parameters:
			bounds.append(par.prior.support)
		if bounds.__len__() > 0:
			self.sampler = SliceSampler(self._parameter_posterior, x0, bounds=bounds, log=True)
			self.marginalize = True
		else:
			self.sampler = None
			self.marginalize = False

	def optimize(self, bounds, iters=20, finish=True, verbose=False, tol=1e-3, num_samples=10, marginalize=True):

		if finish:
			finish = lambda f, x0, **kwargs: fmin_slsqp(f, x0, bounds=bounds, **kwargs)
		else:
			finish = None

		# determine whether or not to marginalize kernel hyperparameters
		marginalize &= self.marginalize

		if marginalize:
			optfun = lambda x, *args: -self._integrated_expected_improvement(x, num_samples)
		else:
			optfun = lambda x, *args: -self._expected_improvement(x)

		for i in range(iters):
			optval = brute(optfun, bounds, finish=finish)
			yield self.X, self.Y, optval
			self.add_observation(optval, self.optfun(optval), tol)

	def set_kernel_parameters(self, values):
		self._recompute = True
		self.kernel.set_parameters(values)

	def add_observation(self, X_new, Y_new, tol):
		X_new, Y_new = format_data(X_new), format_data(Y_new)
		if np.sqrt(np.sum((self.X - X_new)**2, axis=1)).min() > tol:
			self.X = np.concatenate((self.X, X_new), axis=0)
			self.Y = np.concatenate((self.Y, Y_new), axis=0)
			if self.marginalize and self.sampler is not None:
				_ = self.sampler.sample(self.sampler.burnin) # re-sample
			self._recompute = True


	def _gp_posterior(self, x):
		'''
		Using Algorithm 2.1 from Gaussian Processes for Machine Learning.

		Returns:
		mean, variance
		'''
		X, Y, noise, kernel = self.X, self.Y, self.noise, self.kernel

		if self._recompute:
			self._compute_aux()
		L, lower, alpha, ll = self._L, self._lower, self._alpha, self._ll

		K_x = kernel.compute_covariance(x, X)
		v = linalg.solve_triangular(L, K_x, lower=lower)

		mean = K_x.transpose().dot(alpha)
		var = kernel.compute_covariance(x, x) - v.transpose().dot(v)

		return mean[0,0], var[0,0]

	def _gp_loglikelihood(self):		
		if self._recompute:
			self._compute_aux()
		return self._ll

	def _parameter_posterior(self, values):
		'''
		Compute the parameter posterior in order to marginalize the 
		kernel hyperparameters
		'''
		self.set_kernel_parameters(values)
		return self._gp_loglikelihood() + self.kernel.logprior()

	def _expected_improvement(self, x):
		y_best = np.max(self.Y)
		mu, var = self._gp_posterior(x)
		gamma = (y_best - mu)/ np.sqrt(var)
		return np.sqrt(var) * (gamma * norm.cdf(gamma) + norm.pdf(gamma))

	def _integrated_expected_improvement(self, x, num_samples):
		'''
		Marginalize kernel hyperparameters
		'''
		val = 0.
		hp_samples = self.sampler.sample(num_samples)
		for i in range(num_samples):
			self.set_kernel_parameters(hp_samples[i])
			val += self._expected_improvement(x) / num_samples
		return val

	def _compute_aux(self):
		X, Y, noise, kernel = self.X, self.Y, self.noise, self.kernel
		self._L, self._lower = linalg.cho_factor(kernel.compute_covariance(X, X) + noise * np.eye(X.shape[0]), lower=True)
		self._alpha = linalg.cho_solve((self._L, self._lower), Y)
		self._ll = -0.5 * Y.transpose().dot(self._alpha) - np.log(np.diag(self._L)).sum() - float(X.shape[0])/2 * np.log(2 * np.pi) # marginal log likelihood
		self._recompute = False
		return


