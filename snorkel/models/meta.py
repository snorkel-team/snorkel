import os
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# We initialize the engine within the models module because models' schema can depend on
# which data types are supported by the engine
DBURL = os.environ.get('SNORKELDB', 'sqlite:///')
DBNAME = os.environ.get('SNORKELDBNAME', '')

snorkel_postgres = DBURL.startswith('postgres')
if snorkel_postgres:
    connection = DBURL.rstrip('/') + '/' + DBNAME if DBNAME else DBURL
else:
    connection = DBURL + (DBNAME if DBNAME else 'snorkel.db')

snorkel_engine = create_engine(connection)

SnorkelSession = sessionmaker(bind=snorkel_engine)

SnorkelBase = declarative_base(name='SnorkelBase', cls=object)


def clear_database():
    """
    Drop all tables in database.
    Useful before starting a fresh run to avoid conflicts.
    """
    metadata = MetaData(bind=snorkel_engine, reflect=True)
    metadata.drop_all()
    metadata.create_all()
