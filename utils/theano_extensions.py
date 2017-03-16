import theano
import theano.tensor as T
import numpy as np

class ArgminUnique(theano.Op):
	
	def __init__(self, unique, axis, min=True):
		self.unique = unique
		self.axis = axis
		self.min = min
	
	def make_node(self, x):
		x = theano.tensor.as_tensor_variable(x)
		inputs = [x]
		broadcastable = [b for i, b in enumerate(x.type.broadcastable) if i not in [self.axis]]
		outputs = [T.tensor('int64', broadcastable, name='argmax_unique')]
		return theano.Apply(self, inputs, outputs)
	
	def perform(self, node, inputs, outputs):
		x = inputs[0]
		z = outputs[0]
		z[0] = self._fill(x, self.unique, self.axis, self.min)
	
	def infer_shape(self, node, shapes):
		ishape = shapes[0]
		val = tuple([ishape[i] for (i, b) in enumerate(node.inputs[0].type.broadcastable) if i not in [self.axis]])
		return [val]

	def grad(self, inputs, grads):
		x, axis = inputs
		axis_grad = theano.gradient.grad_undefined(self, 1, axis, "ArgminUnique Op doesn't have a gradient w.r.t. to its parameters.")
		return [x.zeros_like(), axis_grad]

	def _fill(self, x, unique, axis, min):
		dims = [d for i, d in enumerate(x.shape) if i != axis]
		indices = [np.arange(d, dtype=np.int32) for d in dims]
		indices = np.meshgrid(*indices)
		indices = np.concatenate([ind.reshape((-1,1)) for ind in indices], axis=1)
		indices = np.concatenate((indices[:,:axis], np.zeros((indices.shape[0],1), dtype=np.int32), indices[:,axis:]), axis=1)

		ret_val = np.zeros(dims, dtype=np.int64)

		dims = [d for i, d in enumerate(x.shape) if i not in [axis, unique]]
		taken_values = [[] for i in range(np.prod(dims))]
	
		if min:
			comp = lambda a,b: a < b
		else:
			comp = lambda a,b: b < a

		for i in range(indices.shape[0]):

			best_idx = -1
			best = np.inf
			if not min:
				best *= (-1)

			index = np.copy(indices[i,:])
			index_insert = np.delete(index, axis)
			index_flat = np.ravel_multi_index(
				np.delete(index, [axis, unique]),
				dims
			)

			for j in range(x.shape[axis]):
				index[axis] = j

				if comp(x[tuple(index)], best) and j not in taken_values[index_flat]:
					best_idx = j
					best = x[tuple(index)]

			# store value
			if best_idx > -1:
				taken_values[index_flat].append(best_idx)

			ret_val[tuple(index_insert)] = best_idx

		return ret_val
					

def argmin_unique(x, unique, axis, min=True):
	return FillOp(unique, axis, min)(x)