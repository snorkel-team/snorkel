from .meta import SnorkelBase
import re
from sqlalchemy import Column, String, Integer, Float, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref


def camel_to_under(name):
    """
    Converts camel-case string to lowercase string separated by underscores.

    Written by epost
    (http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case).

    :param name: String to be converted
    :return: new String with camel-case converted to lowercase, underscored
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


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
                             name='unique_key_candidate_pair'),
        )

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.key.name) + " = " + str(self.value) + ")"


class LabelKey(AnnotationKeyMixin, SnorkelBase):
    pass


class Label(AnnotationMixin, SnorkelBase):
    value = Column(Integer, nullable=False)


class FeatureKey(AnnotationKeyMixin, SnorkelBase):
    pass


class Feature(AnnotationMixin, SnorkelBase):
    value = Column(Float, nullable=False)


class PredictionKey(AnnotationKeyMixin, SnorkelBase):
    pass


class Prediction(AnnotationMixin, SnorkelBase):
    value = Column(Float, nullable=False)
