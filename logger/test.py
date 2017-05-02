import pdb
import time
import numpy as np

import learning_objects as lo
import experiments as exp

class TestLearningObject(lo.BaseLearningObject):
	def __init__(self):
		self._iter = 0
		self.weights = np.random.rand(10,10)
	def train(self, **kwargs):
		self._iter += 1
		# create data to test early stopping
		return np.random.rand(2)
	def get_weights(self):
		return self.weights
	def get_hyperparameters(self):
		return {
			'p': .5,
			'l': 1e-5
		}
	def get_architecture(self):
		return {
			'layer1': 'asdfa',
			'layer2': 'blhrjjah',
			'layer3': 'hjfdd',
			'layer4': 'another thing'
		}
	def load_model(self, weights):
		self.weights = 2 * weights

if __name__ == '__main__':
	e = exp.BaseExperiment(TestLearningObject(), store_weights_in_db=True)

	print('Beginning training (doing stops/starts)...')
	e.train(10)
	e.train(10, begin_new=False)
	e.train(10, begin_new=False)
	e.train(10, begin_new=False)

	train_err = e.get_train_error()
	test_err = e.get_test_error()

	# fig = plt.figure(figsize=(12,6))
	# plt.subplot(2,1,1)
	# plt.plot(train_err)
	# plt.subplot(2,1,2)
	# plt.plot(test_err)
	# plt.show()

	print('New experiment.')
	e.train(10, begin_new=True)

	train_err = e.get_train_error()
	# fig = plt.figure(figsize=(6,6))
	# plt.plot(train_err)
	# plt.show()

	print('Doing new experiment, please interrupt this one.')
	e.train(1000, begin_new=True)
	train_err = e.get_train_error()
	# fig = plt.figure(figsize=(6,6))
	# plt.plot(train_err)
	# plt.show()

	print('restarting same experiment.')
	e.train(10, begin_new=False)
	train_err = e.get_train_error()
	# fig = plt.figure(figsize=(6,6))
	# plt.plot(train_err)
	# plt.show()

	print('Done tests...')

	print("Before loading:")
	print e.learning_object.weights
	e.load_model()
	print("After loading:")
	print e.learning_object.weights

# print 'Done loading model'