import os
import getpass
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

# We initialize the engine within the models module because models' schema can depend on
# which data types are supported by the engine
DBURL = os.environ.get('SNORKELDB', 'sqlite:///')
DBPORT = os.environ.get('SNORKELDBPORT', '5432')
DBUSER = os.environ.get('SNORKELDBUSER', getpass.getuser())
DBNAME = os.environ.get('SNORKELDBNAME', 'snorkel.db')
# Use different default for postgres
PSQLDBNAME = os.environ.get('SNORKELDBNAME', '')

snorkel_postgres = DBURL.startswith('postgres')
if snorkel_postgres:
    # Supports monolithic URL for unix socket connection for postgres
    connection = DBURL.rstrip('/') + '/' + PSQLDBNAME if PSQLDBNAME else DBURL
    connect_args={'sslmode':'disable'}
else:
    connection = DBURL + DBNAME
    connect_args={}

snorkel_engine = create_engine(connection, connect_args = connect_args)

SnorkelSession = scoped_session(sessionmaker(bind=snorkel_engine))

SnorkelBase = declarative_base(name='SnorkelBase', cls=object)


def clear_database():
    """
    Drop all tables in database.
    Useful before starting a fresh run to avoid conflicts.
    """
    metadata = MetaData(bind=snorkel_engine, reflect=True)
    metadata.drop_all()
    metadata.create_all()
