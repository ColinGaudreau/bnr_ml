from sqlalchemy import create_engine, ForeignKey, Column, Integer, String
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.types import JSON, PickleType, DateTime, Float
import cPickle as pickle
import datetime
import sys
from settings import SETTINGS
import os
import paramiko

import pdb

default_settings = {
	'db': 'postgresql+psycopg2',
	'name': 'testdb2',
	'host': '130.179.130.0',
	'user': 'colingaudreau',
	'password': 'Ee45dij7',
	'sftp_password': 'JYRq7Pau', # password for hosting maching
    'weights_dir': '/Users/colingaudreau/.bnr_ml' # assumed to be on same machine as db
}

this = globals()

def create_database_connection(settings):
	if 'db' not in settings:
		raise Exception('blah')
	if 'name' not in settings:
		raise Exception('blah')
	if 'user' not in settings:
		raise Exception('blah')
	db = settings['db']
	name = settings['name']
	user = settings['user']
	if 'host' not in settings:
		host = 'localhost'
	else:
		host = settings['host']
	if 'port' not in settings:
		port = '5432'
	else:
		port = str(settings['host'])

	conn_str = db + '://' + user
	if 'password' in settings and settings['password'] is not None:
		conn_str += ":" + settings['password']

	conn_str += '@' + host + ':' + port + '/' + name

	return create_engine(conn_str)

def _sftp_remove(sftp, filename):
	try:
		sftp.remove(filename)
	except IOError:
		pass

def get_weight_file(experiment, settings):
	weight_dir = settings['weights_dir']
	if weight_dir[-1] != '/':
		weight_dir += '/'
	return weight_dir + settings['name'] + '/experiment_{}_weights.bnr'.format(experiment.id)

def save_weights(experiment, weights, settings):
	weight_dir = settings['weights_dir']
	if weight_dir[-1] != '/':
		weight_dir += '/'
	weight_file = weight_dir + settings['name'] + '/experiment_{}_weights.bnr'.format(experiment.id)
	tmp_weight_file = weight_dir + settings['name'] + '/_tmp_experiment_{}_weights.bnr'.format(experiment.id)

	# connect to server using sftp
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	ssh.connect(settings['host'], username=settings['user'], password=settings['sftp_password'])
	sftp = ssh.open_sftp()

	# create directory for specific database
	sftp.chdir(weight_dir)
	dirs = sftp.listdir()
	if settings['name'] not in dirs:
		sftp.mkdir('./' + settings['name'])

	try:
		with sftp.open(tmp_weight_file, 'wb') as f:
			pickle.dump(weights, f)
		# in case it crashes mid save
		_sftp_remove(sftp, weight_file)
		sftp.rename(tmp_weight_file, weight_file)
	except Exception as e:
		_sftp_remove(sftp, tmp_weight_file) # make sure to clean up if there's an error
		raise e

	ssh.close()
	return

def load_weights(experiment, settings, weights_in_db=False):
	if experiment.weights is not None and weights_in_db:
		return pickle.loads(experiment.weights)

	weight_dir = settings['weights_dir']
	if weight_dir[-1] != '/':
		weight_dir += '/'
	weight_file = 'experiment_{}_weights.bnr'.format(experiment.id)

	weights = None

	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	ssh.connect(settings['host'], username=settings['user'], password=settings['sftp_password'])
	sftp = ssh.open_sftp()

	# create directory for specific database
	sftp.chdir(weight_dir)
	if settings['name'] not in sftp.listdir():
		sftp.mkdir('./' + settings['name'])

	# get into database-specific directory
	sftp.chdir(settings['name'])

	# check if file exists, if so load it
	if weight_file in sftp.listdir():
		with sftp.open(weight_dir + settings['name'] + '/' + weight_file, 'rb') as f:
			weights = pickle.load(f)

	ssh.close()
	return weights

def load_tables(settings=default_settings):
	'''
	This attaches the correct version the Experiment and TrainingResult tables to the module
	'''
	engine = get_engine(settings)
	base = automap_base()
	base.prepare(engine, reflect=True)
	tr = base.classes.training_results
	e = base.classes.experiments
	this['TrainingResult'], this['Experiment'] =  tr, e

def get_engine(settings=default_settings):
	engine = create_database_connection(settings)
	return engine

def create_session(settings=default_settings):
	engine = get_engine(settings)
	Session = sessionmaker()
	Session.configure(bind=engine)
	return Session()

