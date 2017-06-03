from snorkel.models.context import Context
from sqlalchemy import Column, String, Integer, ForeignKey

class RawText(Context):
    """
    A root Context.
    """
    __tablename__ = 'raw_text'
    id = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)
    name = Column(String, unique=True, nullable=False)
    text = Column(String, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'raw_text',
    }

    def get_parent(self):
        return None

    def get_children(self):
        return None

    def __repr__(self):
        return "Raw Text " + str(self.text)
    
# Adds the corresponding table to the underlying database's schema
from snorkel.models.meta import SnorkelBase, snorkel_engine
SnorkelBase.metadata.create_all(snorkel_engine)