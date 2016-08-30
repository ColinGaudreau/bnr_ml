import theano
from theano import tensor as T

def weighted_categorical_crossentropy(predictions, targets, weights):
	'''
	Same as the normal categorical crossentropy, except you can control the FPR for classes.
	This is important for the detector as we can create networks with a high FPR to reject
	obvious negative examples.

	Parameters:
	-------
		predictions:
			This should be a NxM matrix where N is the number of examples and M is the number of
		classes.  The rows should sum to 1.

		targets:
			Same size, save that the rows should only have one 1 indicating the class 
		membership.

		weights:
			Indicate the weighting across classes, ideally sums to near one, but doesn't really
		matter as long as it doesn't get so large as to cause numerical error.
	'''
	loss = targets * T.log(predictions) * weights.dimshuffle('x',0)
	loss = T.sum(loss, axis=1)
	return -T.mean(loss)
