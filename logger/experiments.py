import db
import numpy as np
import learning_objects
import sqlalchemy
import cPickle as pickle

import pdb

class BaseExperiment(object):
	def __init__(
			self,
			learning_object,
			db_settings=db.default_settings,
			use_db=True,	
			store_weights_in_db=False
		):
		assert(issubclass(learning_object.__class__, learning_objects.BaseLearningObject))
		self.learning_object = learning_object
		self.use_db = use_db
		self.store_weights_in_db = store_weights_in_db
		self.iteration = 0
		if use_db:
			self._session = db.create_session(db_settings)
			self.db_settings = db_settings
			# load newest version of tables into memory
			db.load_tables(db_settings)
			self.experiment = db.Experiment()
			self._session.add(self.experiment)

	def train(
			self,
			iterations,
			settings=learning_objects.BaseLearningSettings(),
			save_weights_every=10, 
			note="",
			early_stopping=True,
			stopping_window=10,
			begin_new=False,
			*args,
			**kwargs
		):
		'''
		Train the learning object, save the train/test error every iteration.  Every few
		iterations save the weights of the network.

		Parameters:
		----------
		iterations - Number of iterations for which to train.
		save_weights_every - Save weights every x iterations. 
		'''
		
		self.learning_object.settings = settings
		train_errors, test_errors = [], []

		if self.use_db:
			if begin_new:
				self.experiment = db.Experiment(
						hyperparameters=self.learning_object.get_hyperparameters(),
						architecture=self.learning_object.get_architecture(),
						note=note
					)
				self._session.add(self.experiment)
				self.iteration = 0
			else:
				self.experiment.hyperparameters = self.learning_object.get_hyperparameters()
				self.experiment.architecture = self.learning_object.get_architecture()
				self.experiment.note = note
			self._session.commit()
			if not self.store_weights_in_db:
				self.experiment.weight_file = db.get_weight_file(self.experiment, self.db_settings) # gets id after first commit to db


		try:
			for i in range(self.iteration, self.iteration + iterations):
				train_error, test_error = self.learning_object.train(*args, **kwargs)
				train_errors.append(train_error); test_errors.append(test_error)

				# if train/test error is nan then quit
				if np.isnan(train_error) or np.isnan(test_error):
					print('NaN encountered, stopping training.')
					break

				# save error and save weights
				if self.use_db:
					new_result = db.TrainingResult(train_error=train_error, test_error=test_error, iteration=i, experiment_id=self.experiment.id)
					self._session.add(new_result)
					self._session.commit()

					# save weights
					if ((i - self.iteration) % save_weights_every == 0) or (i - self.iteration) == iterations - 1:
						self.save_experiment()

					self._session.commit()

				# early stopping
				if early_stopping and (i - iterations) > 2 * stopping_window:
					curr_idx = i - iterations
					prev_window = np.mean(test_errors[curr_idx - 2 * stopping_window:curr_idx - stopping_window])
					curr_window = np.mean(test_errors[curr_idx - stopping_window:curr_idx])
					if prev_window < curr_window: # assume problem is minimizing the objective function
						break

		except KeyboardInterrupt:
			if self.use_db:
				self._session.rollback()
				self.save_experiment()

		self.iteration += i
		return np.asarray(train_errors), np.asarray(test_errors)

	def get_train_error(self):
		results = self._session.query(db.TrainingResult).filter(db.TrainingResult.experiment_id == self.experiment.id).all()
		return np.asarray([r.train_error for r in results])

	def get_test_error(self):
		results = self._session.query(db.TrainingResult).filter(db.TrainingResult.experiment_id == self.experiment.id).all()
		return np.asarray([r.test_error for r in results])

	def load_experiment(self, experiment_id, load_model=True):
		'''
		Load experiment, and set iteration to latest one.  Load weights into learning object unless
		otherwise specified.
		'''
		self.experiment = self._session.query(db.Experiment).filter(db.Experiment.id == experiment_id).first()
		last_result = self._session.query(db.TrainingResult).filter(db.TrainingResult.experiment_id == self.experiment.id).order_by(sqlalchemy.desc(db.TrainingResult.iteration)).first()
		if last_result is None:
			self.iteration = 0
		else:
			self.iteration = last_result.iteration + 1
		
		if load_model:
			self.load_model()

	def save_experiment(self):
		weights = self.learning_object.get_weights()
		if self.store_weights_in_db:
			self.experiment.weights = pickle.dumps(weights, protocol=pickle.HIGHEST_PROTOCOL)
			self._session.commit()
		else:
			db.save_weights(self.experiment, self.learning_object.get_weights(), self.db_settings)

	def load_model(self, experiment_id=None):
		'''
		Load weights into learning_object.  Can choose weights from another experiment.
		'''
		experiment = self.experiment
		if experiment_id != None:
			experiment = self._session.query(db.Experiment).filter(db.Experiment.id == experiment_id).first()
		weights = db.load_weights(experiment, self.db_settings, weights_in_db=True)
		if weights is not None:
			self.learning_object.load_model(weights)


