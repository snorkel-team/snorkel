import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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


def new_sessionmaker():
    if 'SNORKELDB' in os.environ and os.environ['SNORKELDB'] != '':
        snorkel_postgres = os.environ['SNORKELDB'].startswith('postgres')
        snorkel_engine = create_engine(os.environ['SNORKELDB'])
    else:
        snorkel_postgres = False
        snorkel_engine = create_engine('sqlite:///snorkel.db')

    SnorkelSession = sessionmaker(bind=snorkel_engine)
    return SnorkelSession
