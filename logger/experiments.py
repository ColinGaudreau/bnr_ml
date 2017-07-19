import db
import numpy as np
import learning_objects
import sqlalchemy
import cPickle as pickle

import pdb

class BasicExperiment(object):
	'''
	Object which manages loading, storing models. Keeping track of experiment hyperparametes.  Saving training information.

	Parameters:
	learning_object : :class:`bnr_ml.logger.learning_objects.BaseLearningObject` instance
		Class representing model that's used.
	db_setting : dict
		Settings dictionary for database.
	store_weights_in_db : bool (default True)
		Whether to store model weights directly in database, or store it in a hidden file and store the file location in the database.
	'''
	def __init__(
			self,
			learning_object,
			db_settings=db.default_settings,
			use_db=True,	
			store_weights_in_db=True
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

		Parameterss
		----------
		iterations : int
			Number of training iterations
		settings : :class:`bnr_ml.logger.learning_objects.BaseLearningSettings` instance
			Settings for training -- includes hyperparameters, and anything that might want to be stored in the database.
		save_weights_every : int (default 10)
			Save the model weights in the database ever specified iterations.
		note : str (default '')
			Note to store in the database before training; could be some general description of the experiment.
		early_stopping : bool (default True)
			Not function yet.
		stopping_window : int (default 10)
			Has to do with `early_stopping`; not functional yet.
		begin_new : bool (default False)
			Whether to create new databaes entry or keep old one -- may be useful if you're restarting training from scratch.
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
				train_ret_args = self.learning_object.train(*args, **kwargs)
				train_error, test_error, extra_info = train_ret_args[0], train_ret_args[1], {}
				if len(train_ret_args) > 2:
					extra_info = train_ret_args[2]
				train_errors.append(train_error); test_errors.append(test_error)

				# if train/test error is nan then quit
				if np.isnan(train_error) or np.isnan(test_error):
					print('NaN encountered, stopping training.')
					break

				# save error and save weights
				if self.use_db:
					new_result = db.TrainingResult(train_error=train_error, test_error=test_error, iteration=i, extra_info=extra_info, experiment_id=self.experiment.id)
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
		'''Get training error from database.'''
		results = self._session.query(db.TrainingResult).filter(db.TrainingResult.experiment_id == self.experiment.id).all()
		return np.asarray([r.train_error for r in results])

	def get_test_error(self):
		'''Get test error from database.'''
		results = self._session.query(db.TrainingResult).filter(db.TrainingResult.experiment_id == self.experiment.id).all()
		return np.asarray([r.test_error for r in results])

	def load_experiment(self, experiment_id, load_model=True):
		'''
		Load experiment, and set iteration to latest one.  Load weights into learning object unless
		otherwise specified.

		Parameters
		----------
		experiment_id : int
			id in database of experiment you're wanting to load.
		load_model : bool (default True)
			Set weights in the model from the weights in the database.
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
		'''
		Save experiment in database.
		'''
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
		self.experiment = experiment
		if weights is not None:
			self.learning_object.load_model(weights)

	def delete(self):
		'''Delete database entry for current experiment -- includes all the TrainingResult instances.'''
		if self.experiment is not None:
			results = self._session.query(db.TrainingResult).filter(db.TrainingResult.experiment_id == self.experiment.id).all()
			for result in results:
				self._session.delete(result)
			self._session.delete(self.experiment)
			self._session.commit()
			self.experiment = None


