class AbstractDetector(object):
	'''
	Abstract class for detectors
	'''
	def detect(self, im, *args, **kwargs):
		raise NotImplementedError()

	def train(self, *args, **kwargs):
		raise NotImplementedError()