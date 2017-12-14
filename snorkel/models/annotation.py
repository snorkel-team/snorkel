from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from sqlalchemy import Column, String, Integer, Float, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref

from .meta import SnorkelBase
from ..utils import camel_to_under


class AnnotationKeyMixin(object):
    """
    Mixin class for defining annotation key tables.
    An AnnotationKey is the unique name associated with a set of Annotations, corresponding e.g. to a single
    labeling or feature function.  An AnnotationKey may have an associated weight (Parameter) associated with it.
    """
    @declared_attr
    def __tablename__(cls):
        return camel_to_under(cls.__name__)

    @declared_attr
    def id(cls):
        return Column(Integer, primary_key=True)

    @declared_attr
    def name(cls):
        return Column(String, nullable=False)

    @declared_attr
    def group(cls):
        return Column(Integer, nullable=False, default=0)

    @declared_attr
    def __table_args__(cls):
        return (UniqueConstraint('name', 'group'),)

    def __repr__(self):
        return str(self.__class__.__name__) + " (" + str(self.name) + ")"


class GoldLabelKey(AnnotationKeyMixin, SnorkelBase):
    pass


class LabelKey(AnnotationKeyMixin, SnorkelBase):
    pass


class FeatureKey(AnnotationKeyMixin, SnorkelBase):
    pass


class PredictionKey(AnnotationKeyMixin, SnorkelBase):
    pass    


class AnnotationMixin(object):
    """
    Mixin class for defining annotation tables. An annotation is a value associated with a Candidate.
    Examples include labels, features, and predictions.
    New types of annotations can be defined by creating an annotation class and corresponding annotation,
    for example:

    .. code-block:: python

        from snorkel.models.annotation import AnnotationMixin
        from snorkel.models.meta import SnorkelBase

        class NewAnnotation(AnnotationMixin, SnorkelBase):
            value = Column(Float, nullable=False)


        # The entire storage schema, including NewAnnotation, can now be initialized with the following import
        import snorkel.models

    The annotation class should include a Column attribute named value.
    """
    @declared_attr
    def __tablename__(cls):
        return camel_to_under(cls.__name__)

    # The key is the "name" or "type" of the Annotation- e.g. the name of a feature, or of a human annotator
    @declared_attr
    def key_id(cls):
        return Column('key_id', Integer, ForeignKey('%s_key.id' % cls.__tablename__, ondelete='CASCADE'), primary_key=True)

    @declared_attr
    def key(cls):
        return relationship('%sKey' % cls.__name__, backref=backref(camel_to_under(cls.__name__) + 's', cascade='all, delete-orphan'))

    # Every annotation is with respect to a candidate
    @declared_attr
    def candidate_id(cls):
        return Column('candidate_id', Integer, ForeignKey('candidate.id', ondelete='CASCADE'), primary_key=True)

    @declared_attr
    def candidate(cls):
        return relationship('Candidate', backref=backref(camel_to_under(cls.__name__) + 's', cascade='all, delete-orphan', cascade_backrefs=False),
                            cascade_backrefs=False)

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.key.name) + " = " + str(self.value) + ")"


class GoldLabel(AnnotationMixin, SnorkelBase):
    """A separate class for labels from human annotators or other gold standards."""
    value = Column(Integer, nullable=False)


class Label(AnnotationMixin, SnorkelBase):
    """
    A discrete label associated with a Candidate, indicating a target prediction value.

    Labels are used to represent the output of labeling functions.

    A Label's annotation key identifies the labeling function that provided the Label.
    """
    value = Column(Integer, nullable=False)


class Feature(AnnotationMixin, SnorkelBase):
    """
    An element of a representation of a Candidate in a feature space.

    A Feature's annotation key identifies the definition of the Feature, e.g., a function that implements it
    or the library name and feature name in an automatic featurization library.
    """
    value = Column(Float, nullable=False)


class Prediction(AnnotationMixin, SnorkelBase):
    """
    A probability associated with a Candidate, indicating the degree of belief that the Candidate is true.

    A Prediction's annotation key indicates which process or method produced the Prediction, e.g., which
    model with which ParameterSet.
    """
    value = Column(Float, nullable=False)


class StableLabel(SnorkelBase):
    """
    A special secondary table for preserving labels created by *human annotators* (e.g. in the Viewer)
    in a stable format that does not cascade, and is independent of the Candidate ids.
    """
    __tablename__  = 'stable_label'
    context_stable_ids = Column(String, primary_key=True)  # ~~ delimited list of the context stable ids
    annotator_name     = Column(String, primary_key=True)
    split              = Column(Integer, default=0)
    value              = Column(Integer, nullable=False)

    def __repr__(self):
        return "%s (%s : %s)" % (self.__class__.__name__, self.annotator_name, self.value)
