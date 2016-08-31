from .meta import SnorkelBase
import re
from sqlalchemy import Column, String, Integer, Float, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref
from snorkel.utils import camel_to_under


# TODO: Switch to AnnotationKey(SnorkelBase)
class AnnotationKeyMixin(object):
    """
    Mixin class for defining annotation key tables.

    Annotation key tables are mappings from unique string names to integer id numbers.
    These strings uniquely identify who or what produced an annotation.
    """
    @declared_attr
    def __tablename__(cls):
        return camel_to_under(cls.__name__)

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    def __repr__(self):
        return str(self.__class__.__name__) + " (" + str(self.name) + ")"


class AnnotationMixin(object):
    """
    Mixin class for defining annotation tables.

    An annotation is a value associated with a Candidate. Examples include labels, features,
    and predictions.

    New types of annotations can be defined by creating an annotation class and corresponding annotation
    key class, for example:
    >>> from snorkel.models.annotation import AnnotationKeyMixin, AnnotationMixin
    >>>
    >>> class NewAnnotationKey(AnnotationKeyMixin, SnorkelBase):
    >>>     pass
    >>>
    >>>
    >>> class NewAnnotation(AnnotationMixin, SnorkelBase):
    >>>     value = Column(Float, nullable=False)
    >>>
    >>>
    >>> # The entire storage schema, including NewAnnotation, can now be initialized with the following import
    >>> import snorkel.models

    Note that the two classes should have the same name, except for 'Key' at the end of the annotation key class.
    The annotation class should include a Column attribute named value.
    """
    @declared_attr
    def __tablename__(cls):
        return camel_to_under(cls.__name__)

    id = Column(Integer, primary_key=True)

    @declared_attr
    def key_id(cls):
        return Column(camel_to_under(cls.__name__) + '_key_id',
                      Integer,
                      ForeignKey(camel_to_under(cls.__name__) + '_key.id'))

    @declared_attr
    def key(cls):
        return relationship(cls.__name__ + 'Key', backref=backref(camel_to_under(cls.__name__) + 's', cascade='all'))

    @declared_attr
    def candidate_id(cls):
        return Column(camel_to_under(cls.__name__) + '_candidate_id', Integer, ForeignKey('candidate.id'))

    @declared_attr
    def candidate(cls):
        return relationship('Candidate', backref=backref(camel_to_under(cls.__name__) + 's', cascade_backrefs=False))

    @declared_attr
    def __table_args__(cls):
        return (
            UniqueConstraint(camel_to_under(cls.__name__) + '_key_id',
                             camel_to_under(cls.__name__) + '_candidate_id',
                             name=camel_to_under(cls.__name__) + '_unique_key_candidate_pair'),
        )

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.key.name) + " = " + str(self.value) + ")"


class LabelKey(AnnotationKeyMixin, SnorkelBase):
    """
    Key for Labels.
    """
    pass


class Label(AnnotationMixin, SnorkelBase):
    """
    A discrete label associated with a Candidate, indicating a target prediction value.

    Labels are used to represent both human-provided annotations and the output of labeling functions.

    A Label's LabelKey identifies the person or labeling function that provided the Label.
    """
    value = Column(Integer, nullable=False)


class FeatureKey(AnnotationKeyMixin, SnorkelBase):
    """
    Key for Features.
    """
    pass


class Feature(AnnotationMixin, SnorkelBase):
    """
    An element of a representation of a Candidate in a feature space.

    A Feature's FeatureKey identifies the definition of the Feature, e.g., a function that implements it
    or the library name and feature name in an automatic featurization library.
    """
    value = Column(Float, nullable=False)


class PredictionKey(AnnotationKeyMixin, SnorkelBase):
    """
    Key for Predictions.
    """
    pass


class Prediction(AnnotationMixin, SnorkelBase):
    """
    A probability associated with a Candidate, indicating the degree of belief that the Candidate is true.

    A Prediction's PredictionKey indicates which process or method produced the Prediction, e.g., which
    model with which ParameterSet.
    """
    value = Column(Float, nullable=False)
