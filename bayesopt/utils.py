import numpy as np
from sobol_seq import i4_sobol

def format_data(x, dim=None):
	if isinstance(x, list):
			x = np.asarray([x])
	elif not isinstance(x, np.ndarray):
		x = np.asarray([[x]])
	elif x.shape.__len__() < 2:
		if dim is None:
			x = x.reshape((1,-1))
		else:
			x = x.reshape((-1, dim))
	return x

def sobol_seq(dim, n, skip=1, seed=0):
	init_seed = seed
	r = np.full((n, dim), np.nan)
	for j in range(n):
		seed = init_seed + j + 1
		r[j, 0:dim], next_seed = i4_sobol(dim, seed)
	return r, next_seed