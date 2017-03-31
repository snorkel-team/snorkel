import os
import getpass
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# We initialize the engine within the models module because models' schema can depend on
# which data types are supported by the engine
DBURL = os.environ.get('SNORKELDB', 'sqlite:///')
DBPORT = os.environ.get('SNORKELDBPORT', '5432')
DBUSER = os.environ.get('SNORKELDBUSER', getpass.getuser())
DBNAME = os.environ.get('SNORKELDBNAME', 'snorkel.db')

snorkel_postgres = DBURL.startswith('postgres')
snorkel_conn_string = ''
connect_args={}
if snorkel_postgres:
    if '///' in DBURL:
        # Supports monolithic URL for unix socket connection for postgres
        snorkel_conn_string = DBURL + DBNAME
    else:
        snorkel_conn_string = DBURL.rstrip('/') + '/' + DBNAME
    #connect_args={'sslmode':'disable'}
else:
    snorkel_conn_string = DBURL + DBNAME

SnorkelBase = declarative_base(name='SnorkelBase', cls=object)

def new_engine():
    return create_engine(snorkel_conn_string, connect_args = connect_args)

def new_session(engine):
    return sessionmaker(bind=engine)

def new_sessionmaker(engine=None):
    if not engine: engine = create_engine(snorkel_conn_string)
    # New sessionmaker
    SnorkelSession = sessionmaker(bind=engine)
    return SnorkelSession

snorkel_engine = new_engine()
SnorkelSession = new_session(snorkel_engine)

def clear_database():
    """
    Drop all tables in database.
    Useful before starting a fresh run to avoid conflicts.
    """
    metadata = MetaData(bind=snorkel_engine, reflect=True)
    metadata.drop_all()
    metadata.create_all()
