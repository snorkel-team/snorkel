import os
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import contextlib

# We initialize the engine within the models module because models' schema can depend on
# which data types are supported by the engine
if 'SNORKELDB' in os.environ and os.environ['SNORKELDB'] != '':
    snorkel_postgres = os.environ['SNORKELDB'].startswith('postgres')
    snorkel_engine = create_engine(os.environ['SNORKELDB'])
else:
    snorkel_postgres = False
    snorkel_engine = create_engine('sqlite:///snorkel.db')

SnorkelSession = sessionmaker(bind=snorkel_engine)

SnorkelBase = declarative_base(name='SnorkelBase', cls=object)

def clear_database():
    '''
    Delete all table contents in the database while keeping the schemas
    Useful before starting a fresh run to avoid constraint violations.
    '''
    metadata = MetaData(bind=snorkel_engine, reflect=True)
    with contextlib.closing(snorkel_engine.connect()) as con:
        trans = con.begin()
        for table in reversed(metadata.sorted_tables):
            con.execute(table.delete())
        trans.commit()