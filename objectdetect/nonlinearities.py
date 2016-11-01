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
	less_than_one = T.abs(val) < 1
	greater_than_one = T.abs(val) >= 1
	cost = T.set_subtensor(val[less_than_one.nonzero()], 0.5 * val[less_than_one.nonzero()]**2)
	cost = T.set_subtensor(val[greater_than_one.nonzero()], T.abs(val[less_than_one.nonzero()]) - 0.5)
	return cost

def safe_sqrt(val, eps=1e-3):
	return T.sqrt(val + eps)