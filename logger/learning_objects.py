import pdb

class BaseLearningSettings(object):
	'''Base class for all learning settings objects.'''
	def serialize(self):
		'''Get dictionary of all the informaton you want stored in the database.'''
		return self.__dict__

class BaseLearningObject(object):
	'''Base class for all learning objects.'''
	def __init__(self):
		self.settings = BaseLearningSettings()
		self._hyperparameters = []

	def train(self, *args, **kwargs):
		'''Train model.'''
		raise NotImplementedError()

	def get_weights(self):
		'''Get weights of model.'''
		raise NotImplementedError()

	def get_hyperparameters(self):
		'''Get hyperparameters of model.'''
		self._hyperparameters.append(self.settings.serialize())
		return self._hyperparameters

	def get_architecture(self):
		'''Get architectrue of model.'''
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
