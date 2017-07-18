import numpy as np
import numpy.random as npr
import scipy.linalg as linalg

import pdb

class BaseSampler(object):
	'''
	Base class for all samplers.
	'''
	def sample(self, N, *args, **kwargs):
		'''
		Sample from the (hopefully) stationary distribution.
		'''
		raise NotImplementedError('sample() must be implemented in child class.')

class SimpleSliceSampler(BaseSampler):
	'''
	Basic slice sampler, uses Gibbs sampling for higher dimensions.
	'''
	def __init__(self, pdf, x0, step_out_width=.05, step_out_method='double', burnin=100, seed=1991, bounds=None, log=True):
		if bounds is None:
			bounds = [(-np.inf, np.inf) for d in range(x0.shape[0])]

		if isinstance(x0, list):
			x0 = np.asarray(x0)
		elif isinstance(x0, float) or isinstance(x0, int):
			x0 = np.asarray([x0])

		if isinstance(x0, np.ndarray):
			step_out_width = step_out_width * np.ones(x0.size)
			assert bounds is None or bounds.__len__() == x0.size

		self.logpdf = pdf if log else lambda x: np.log(pdf(x))
		self._curr_x = x0
		self.step_out_width = step_out_width
		self.rnd_gen = npr.RandomState(seed)
		self.bounds = bounds
		self.burnin = burnin

		self._scale = np.inf
		
		assert step_out_method.lower() == 'double' or step_out_method.lower() == 'increment'
		self.step_out_method = step_out_method.lower()
		
		# do burnin samples
		_ = self.sample(burnin)
	
	def sample(self, N, verbose=False):
		'''
		'''
		new_sample = np.zeros((N, self._curr_x.size))
		
		for i in range(N):
			for j in range(self._curr_x.size): # do gibbs sampling
				y = self.logpdf(self._curr_x) + np.log(self.rnd_gen.rand())
				interval = self._step_out(self._curr_x, y, j)

				# do shrinkage
				valid_sample = False
				while not valid_sample:
					x_new = np.copy(self._curr_x)
					x_new[j] = interval[0] + (interval[1] - interval[0]) * self.rnd_gen.rand()

					if self.logpdf(x_new) > y:
						valid_sample = True
					elif x_new[j] < self._curr_x[j]:
						interval[0] = x_new[j]
					else:
						interval[1] = x_new[j]
				self._curr_x = x_new
			new_sample[i] = x_new

			if verbose:
				print('Sample %d drawn.' % i)
		return np.asarray(new_sample)

	def _validate_interval(self, xi, xf, index):
		if self.bounds is not None:
			if xi[index] < self.bounds[index][0]:
				xi[index] = self.bounds[index][0]
			if xf[index] > self.bounds[index][1]:
				xf[index] = self.bounds[index][1]
		return xi, xf

	def _step_out(self, x, y, index):
		xi, xf = np.copy(x), np.copy(x)
		xi[index], xf[index] = x[index] - self.step_out_width[index], x[index] + self.step_out_width[index]
		xi, xf = self._validate_interval(xi, xf, index)
		while (self.logpdf(xi) > y and self.logpdf(xf) > y) and not (xi[index] == self.bounds[index][0] and xf[index] == self.bounds[index][1]):
			if self.step_out_method == 'increment':
				self.step_out_width[index] += self.step_out_width[index]
			elif self.step_out_method == 'double':
				self.step_out_width[index] *= 2

			xi[index], xf[index] = x[index] - self.step_out_width[index], x[index] + self.step_out_width[index]
			xi, xf = self._validate_interval(xi, xf, index)

		return [xi[index], xf[index]]

class MDSliceSampler(BaseSampler):
	'''
	Multidimensional slice sampler, uses random eigenvectors slices for higher dimensions.

	Parameters
	----------
	pdf : function
		Probability density function -- can be log or not.
	x0 : list or numpy.ndarray
		Initial point for chain.
	width : float (default 1.0)
		Width of initial interval for slice sampler.
	n_cached : int (default 1000)
		Number of point to cache for calculating sample covariance.
	burnin : int (default 1000)
		Number of points for burnin.
	seed : int (default 1991)
		Seed for random number generator.
	bounds : List or None (default None)
		If `None` then no bounds, otherwise bounds on the pdf.
	log : bool (default True)
		Whether `pdf` is the log of the PDF.
	'''
	def __init__(self, pdf, x0, width=1., n_cached=1000, burnin=1000, seed=1991, bounds=None, log=True):
		if bounds is None:
			bounds = [(-np.inf, np.inf) for d in range(x0.shape[0])]
		if isinstance(x0, list):
			x0 = np.asarray(x0)
		elif isinstance(x0, float) or isinstance(x0, int):
			x0 = np.asarray([x0])
		if len(x0.shape) > 1:
			x0 = x0.flatten()

		self.logpdf = pdf if log else lambda x: np.log(pdf(x))
		self._x_curr = x0
		self.width = width
		self.rnd_gen = npr.RandomState(seed)
		self.bounds = bounds
		self.burnin = burnin
		self._n_cached = n_cached
		self._dim = x0.shape[0]

		# define stuff for caching samples to calculate cov
		self._cache_count = 0
		self._cached_samples = np.zeros((n_cached, self._dim))
		self._ready_for_cov = False # enough samples have been collected to start using cov eigenvectors
		self._basis = np.eye(self._dim) # define standard basis

		_ = self.sample(burnin)

	def _point_in_bound(self, vec, dir_vec):
		on_boundary = False
		for d, bound in enumerate(self.bounds):
			if vec[d] < bound[0]:
				vec +=dir_vec * (bound[0] - vec[d]) / dir_vec[d]
				on_boundary = True
			if vec[d] > bound[1]:
				vec += dir_vec * (bound[1] - vec[d]) / dir_vec[d]
				on_boundary = True
		return vec, on_boundary

	def sample(self, N, n_recalc=500, beta=.05):
		'''
		Sampler from the stationary distribution.

		Parameters
		----------
		N : int
			Number of samples.
		n_recalc : int (default 500)
			Number of iterations after which to re-calculate the covariance.
		beta : float (default 0.05)
			Chance that an iteration uses a random slice rather than a random eigenvector.

		Returns
		-------
		numpy.ndarray
			Samples from the chain.
		'''
		width, x_curr, n_cached = self.width, self._x_curr, self._n_cached

		samples = np.zeros((N, self._dim))
		shrinks = []
		for n in range(N):
			if self._ready_for_cov and self.rnd_gen.rand() > beta:
				vec = self._basis[:,self.rnd_gen.randint(self._dim)]
				lb = x_curr - width * vec * self.rnd_gen.rand()
				ub = lb + width * vec
			else:
				rnd_vec = self.rnd_gen.rand(self._dim) - .5
				rnd_vec /= np.sqrt(np.sum(rnd_vec**2))
				lb = x_curr - width * rnd_vec * self.rnd_gen.rand() # slice along random direction
				ub = lb + width * rnd_vec

			y = self.logpdf(x_curr) + np.log(self.rnd_gen.rand())
			lb, ub = self._step_out(x_curr, lb, ub, y)

			sample_found = False
			cnt = 0
			norm = lambda x, y: np.sum((x-y)**2)
			while not sample_found:
				rnd_weight = self.rnd_gen.rand()
				x_new = rnd_weight * lb + (1 - rnd_weight) * ub

				if self.logpdf(x_new) > y:
					x_curr = x_new
					sample_found = True
				else:
					lb, ub = self._shrink(x_curr, x_new, lb ,ub)
				cnt += 1
			shrinks.append(cnt)

			samples[n] = x_curr
			self._cached_samples[self._cache_count % n_cached] = x_curr
			self._cache_count += 1

			if self._cache_count % n_recalc == 0:
				self._ready_for_cov = True
				centered_samples = (self._cached_samples - np.mean(self._cached_samples, axis=0, keepdims=True))
				e, v = linalg.eig(centered_samples.transpose().dot(centered_samples) / centered_samples.shape[0])
				self._sqr_eigs = np.sqrt(np.real(e))
				self._basis = v

		self._x_curr = x_curr
		return samples

	def _shrink(self, x_curr, x_prop, lb, ub):
		'''
		Shrink hyperrectangle in basis of sample covariance eigenvectors
		'''
		norm = lambda x,y: np.sum((x - y)**2)
		if norm(x_prop, lb) < norm(x_curr, lb):
			lb = x_prop
		elif norm(x_prop, ub) < norm(x_curr, ub):
			ub = x_prop
		return lb, ub

	def _step_out(self, x_curr, lb, ub, y, max_iter=5):
		dir_vec = ub - lb / np.sum((ub - lb)**2)
		itr=0
		dir_vec = ub - lb
		lb, on_boundary = self._point_in_bound(lb, dir_vec)
		while (self.logpdf(lb) > y and itr < max_iter) and (not on_boundary):
			lb += 2 * (lb - x_curr)
			lb, on_boundary = self._point_in_bound(lb, dir_vec)
			itr += 1
		itr=0
		ub, on_boundary = self._point_in_bound(ub, dir_vec)
		while (self.logpdf(ub) > y and itr < max_iter) and (not on_boundary):
			ub += 2 * (ub - x_curr)
			ub, on_boundary = self._point_in_bound(ub, dir_vec)
			itr += 1
		return lb, ub



