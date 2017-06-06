"""
Subpackage for all built-in Snorkel data models.

After creating additional models that extend snorkel.models.meta.SnorkelBase, they must be added to the database schema.
For example, the correct way to define a new type of Context is:

.. code-block:: python

    from snorkel.models.context import Context
    from sqlalchemy import Column, String, ForeignKey

    class NewType(Context):
        # Declares name for storage table
        __tablename__ = 'newtype'
        # Connects NewType records to generic Context records
        id = Column(String, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)

        # Polymorphism information for SQLAlchemy
        __mapper_args__ = {
            'polymorphic_identity': 'newtype',
        }

        # Rest of class definition here

    # Adds the corresponding table to the underlying database's schema
    from snorkel.models.meta import SnorkelBase, snorkel_engine
    SnorkelBase.metadata.create_all(snorkel_engine)
"""
from .meta import SnorkelBase, SnorkelSession, snorkel_engine, snorkel_postgres
from .context import Context, Document, Sentence, TemporarySpan, Span
from .context import construct_stable_id, split_stable_id
from .candidate import Candidate, candidate_subclass, Marginal
from .annotation import (
    Feature, FeatureKey, Label, LabelKey, GoldLabel, GoldLabelKey, StableLabel,
    Prediction, PredictionKey
)

# This call must be performed after all classes that extend SnorkelBase are
# declared to ensure the storage schema is initialized
SnorkelBase.metadata.create_all(snorkel_engine)
