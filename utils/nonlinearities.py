import theano
from theano import tensor as T

def softmax(mat, axis=1):
	'''	
	Softmax activation function: :math:`\\frac{\exp(x_j)}{\sum_{i=0} \exp(x_i)}`.

	Parameters
	----------
	mat : theano.tensor
		Theano tensor on which to do the softmax function
	axis : int (default 1)
		Axis along which to perform softmax operation.
	'''
	max_el = mat.max(axis=axis, keepdims=True)
	logsoftmax = mat - (max_el + T.log(T.sum(T.exp(mat - max_el), axis=axis, keepdims=True)))
	return T.exp(logsoftmax)

def smooth_l1(val):
	'''
	Smooth :math:`L_1` function.
	'''
	cost = T.switch(T.abs_(val)<1, 0.5 * val**2, T.abs_(val) - 0.5)
	return cost

def smooth_abs(x, x0=.01):
	'''
	Smooth absolute value function.
	'''
	a1 = 1 / (2 * x0)
	a2 = - 1 / (2 * x0)
	b1 = x0 - a1 * x0**2
	b2 = x0 + a2 * x0**2
	idx1, idx2 = x < -x0, T.abs_(x) <= x0
	val = T.switch(idx1, -x - b2, T.switch(idx2, a1 * x**2, x - b1))
	return val

def safe_sqrt(val, eps=1e-6):
	'''
	Safe square root -- if slightly negative, makes value positive.
	'''
	return T.sqrt(val + eps)
