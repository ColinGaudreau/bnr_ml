from sqlalchemy import ForeignKey, Column, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import JSON, PickleType, DateTime, Float
import cPickle as pickle
import datetime
from migrate import *

Base = declarative_base()

class Experiment(Base):
    '''
    Model for the experiments table
    '''
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    hyperparameters = Column(JSON)
    architecture = Column(JSON)
    weights = Column(PickleType(protocol=pickle.HIGHEST_PROTOCOL, pickler=pickle))
    weight_file = Column(String(length=96))
    note = Column(String)
    began = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return "<Experiment (id=%s), architecture=%s, hyperparameters=%s>" % (self.id, self.architecture, self.hyperparameters)

class TrainingResult(Base):
    '''
    Model for an individual training result
    '''
    __tablename__ = 'training_results'

    id = Column(Integer, primary_key=True)
    train_error = Column(Float(precision=32), default=0)
    # validation_error = Column(Float(precision=32), default=0)
    test_error = Column(Float(precision=32), default=0)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    iteration = Column(Integer)

    experiment = relationship("Experiment", back_populates="results")

    def __repr__(self):
        return "<TrainingResult (experiment=%s), train_error=%s, test_error=%s, iteration=%s>" % (self.experiment.id, self.train_error, self.test_error, self.iteration)

Experiment.results = relationship("TrainingResult", order_by=TrainingResult.id, back_populates="experiment")

def upgrade(migrate_engine):
    # Upgrade operations go here. Don't create your own engine; bind
    # migrate_engine to your metadata
    Experiment.__table__.create(migrate_engine)
    TrainingResult.__table__.create(migrate_engine)

def downgrade(migrate_engine):
    # Operations to reverse the above upgrade go here.
    TrainingResult.__table__.drop(migrate_engine)
    Experiment.__table__.drop(migrate_engine)
