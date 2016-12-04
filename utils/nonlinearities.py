import theano
from theano import tensor as T

def softmax(mat, axis=1):
	'''	
	axis along which to take soft max, axis \in {0,1,2,3}

	Safe softmax function:
	log f(x) = x - (x_1 + log(1 + sum_{i=2}^N exp(x_i - x_1)))
	'''
	max_el = mat.max(axis=axis, keepdims=True)
	logsoftmax = mat - (max_el + T.log(T.sum(T.exp(mat - max_el), axis=axis, keepdims=True)))
	return T.exp(logsoftmax)

def smooth_l1(val):
	cost = T.switch(T.abs_(val)<1, 0.5 * val**2, T.abs_(val) - 0.5)
	return cost

def smooth_abs(val, x0=.1):
	a1 = 1 / (2 * x0)
	a2 = - 1 / (2 * x0)
	b1 = x0 - a1 * x0**2
	b2 = x0 + a2 * x0**2
	idx1, idx2 = x < -x0, T.abs_(x) <= x0
	val = T.switch(idx1, -x - b2, T.switch(idx2, a1 * x**2, x - b1))
	return val

def safe_sqrt(val, eps=1e-3):
	return T.sqrt(val + eps)
