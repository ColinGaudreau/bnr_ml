import theano
from theano import tensor as T

def sgd(cost, params, lr=1e-3):
	lr = T.cast(lr, dtype=theano.config.floatX)

	gparams = T.grad(cost, params)
	updates = OrderedDict()
 
	for param, gparam in zip(params, gparams):
		updates[param] = param - lr * gparam

	return updates

def momentum(cost, params, lr=1e-4, momentum=.9):
	momentum = T.cast(momentum, theano.config.floatX)
	lr = T.cast(lr, dtype=theano.config.floatX)

	gparams = T.grad(cost, params)
	updates = OrderedDict()

	for param, gparam in zip(params, gparams):
		velocity = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=value.dtype), broadcastable=param.broadcastable)
		updates[velocity] = momentum - lr * gparam
		updates[param] = param + updates[velocity]

	return updates
