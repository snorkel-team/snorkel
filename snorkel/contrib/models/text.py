from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from snorkel.models import Context
from snorkel.models.meta import SnorkelBase, snorkel_engine
from sqlalchemy import Column, String, Integer, Text, ForeignKey, UniqueConstraint


class RawText(Context):
    """A simple Context class representing a blob of text."""
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
SnorkelBase.metadata.create_all(snorkel_engine)