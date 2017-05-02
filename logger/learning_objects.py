import pdb

class BaseLearningSettings(object):
	def serialize(self):
		return self.__dict__

class BaseLearningObject(object):
	def __init__(self):
		self.settings = BaseLearningSettings()
		self._hyperparameters = []

	def train(self, *args, **kwargs):
		raise NotImplementedError()

	def get_weights(self):
		'''
		This should return weights
		'''
		raise NotImplementedError()

	def get_hyperparameters(self):
		'''
		This should return a dictionary of hyperparameters
		'''
		self._hyperparameters.append(self.settings.serialize())
		return self._hyperparameters

	def get_architecture(self):
		'''
		This should return a dictionary describing the network architecture
		'''
		raise NotImplementedError

	def load_model(self, weights):
		'''
		This returns some sort of model from a buffer holding the weights
		'''
		raise NotImplementedError

	def __setattr__(self, name, value):
		if name != 'settings':
			return super(BaseLearningObject, self).__setattr__(name, value)
		else:
			return self._set_settings(value)

	def _set_settings(self, s):
		assert(issubclass(s.__class__, BaseLearningSettings))	
		return super(BaseLearningObject, self).__setattr__('settings', s)
