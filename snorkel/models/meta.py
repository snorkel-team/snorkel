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


class SnorkelComparable(object):
    def __hash__(self):
        raise NotImplementedError('Classes extending SnorkelBase must implement __eq__, __ne__, and __hash__.')

    def __eq__(self):
        raise NotImplementedError('Classes extending SnorkelBase must implement __eq__, __ne__, and __hash__.')

    def __ne__(self):
        raise NotImplementedError('Classes extending SnorkelBase must implement __eq__, __ne__, and __hash__.')

    def __le__(self):
        raise NotImplementedError('Classes extending SnorkelBase do not support inequality comparisons by default.')

    def __lt__(self):
        raise NotImplementedError('Classes extending SnorkelBase do not support inequality comparisons by default.')

    def __gt__(self):
        raise NotImplementedError('Classes extending SnorkelBase do not support inequality comparisons by default.')

    def __ge__(self):
        raise NotImplementedError('Classes extending SnorkelBase do not support inequality comparisons by default.')


SnorkelBase = declarative_base(name='SnorkelBase', cls=SnorkelComparable)

