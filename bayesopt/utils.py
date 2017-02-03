import numpy as np

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