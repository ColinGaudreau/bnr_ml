import numpy as np
import numpy.random as npr

import pdb

class SliceSampler(object):
	'''
	Very basic slice sampler
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

		if not log:
			self.logpdf = lambda x: np.log(pdf(x))
		else:
			self.logpdf = pdf
		self._curr_x = x0
		self.step_out_width = step_out_width
		self.rnd_gen = npr.RandomState(seed)
		self.bounds = bounds
		self.log = log
		self.burnin = burnin

		self._scale = np.inf
		
		assert step_out_method.lower() == 'double' or step_out_method.lower() == 'increment'
		self.step_out_method = step_out_method.lower()
		
		# do burnin samples
		_ = self.sample(burnin)
	
	def sample(self, N, verbose=False):
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
