from sqlalchemy import Table, Column, MetaData
from sqlalchemy.types import Float, ARRAY
import pickle
import datetime
from migrate import *

def upgrade(migrate_engine):
    # Upgrade operations go here. Don't create your own engine; bind
    # migrate_engine to your metadata
    meta = MetaData(bind=migrate_engine)
    training_results = Table('training_results', meta, autoload=True)
    column = Column('extra_info', ARRAY(Float(precision=32)))
    column.create(training_results)

def downgrade(migrate_engine):
    # Operations to reverse the above upgrade go here.
    meta = MetaData(bind=migrate_engine)
    training_results = Table('training_results', meta, autoload=True)
    column = Column('extra_info', ARRAY(Float(precision=32)))
    column.drop(training_results)