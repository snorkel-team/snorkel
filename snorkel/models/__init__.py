"""
Subpackage for all built-in Snorkel data models.

To ensure correct behavior, this subpackage should always be treated as a single module (with one exception
described below). This rule means that all data models should be imported from this subpackage,
not directly from individual submodules. For example, the correct way to import the Corpus class is

.. code-block:: python

    from snorkel.models import Corpus

The only exception is importing SnorkelBase or other classes in order to extend Snorkel's data models.
To ensure that any additional data models are included in the storage backend, these must be imported
and the extending subtypes defined before importing `snorkel.models`. For example, the correct way to
define a new type of Context is:

.. code-block:: python

    from snorkel.models.context import Context
    from sqlalchemy import Column, String, ForeignKey

    class NewType(Context):
        # Declares name for storage table
        __tablename__ = 'newtype'
        # Connects NewType records to generic Context records
        id = Column(String, ForeignKey('context.id'))

        # Polymorphism information for SQLAlchemy
        __mapper_args__ = {
            'polymorphic_identity': 'newtype',
        }

        # Rest of class definition here

    # The entire storage schema, including NewType, can now be initialized with the following import
    import snorkel.models
"""
from .meta import SnorkelBase, SnorkelSession, snorkel_engine, snorkel_postgres
from .context import Context, Corpus, Document, Sentence, TemporarySpan, Span
from .context import construct_stable_id, split_stable_id
from .candidate import Candidate, CandidateSet, candidate_subclass
from .annotation import Feature, Label, Prediction, AnnotationKey, AnnotationKeySet
from .parameter import Parameter, ParameterSet

# This call must be performed after all classes that extend SnorkelBase are
# declared to ensure the storage schema is initialized
SnorkelBase.metadata.create_all(snorkel_engine)
