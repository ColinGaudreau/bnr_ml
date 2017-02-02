import numpy as np

def format_data(x):
	if isinstance(x, list):
			x = np.asarray([x])
	elif not isinstance(x, np.ndarray):
		x = np.asarray([[x]])
	elif x.shape.__len__() < 2:
		x = x.reshape((-1,1))
	return x