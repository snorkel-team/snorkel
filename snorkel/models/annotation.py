from .meta import SnorkelBase
import re
from sqlalchemy import Column, String, Integer, Float, ForeignKey, UniqueConstraint, Table
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref
from snorkel.utils import camel_to_under
from sqlalchemy.orm.collections import attribute_mapped_collection


annotation_key_set_annotation_key_association = \
    Table('annotation_key_set_annotation_key_association', SnorkelBase.metadata,
          Column('annotation_key_set_id', Integer, ForeignKey('annotation_key_set.id')),
          Column('annotation_key_id', Integer, ForeignKey('annotation_key.id')))


class AnnotationKeySet(SnorkelBase):
    """A many-to-many set of AnnotationKeys."""
    __tablename__ = 'annotation_key_set'
    id            = Column(Integer, primary_key=True)
    name          = Column(String, unique=True, nullable=False)
    keys          = relationship('AnnotationKey',
                                 secondary=annotation_key_set_annotation_key_association,
                                 collection_class=attribute_mapped_collection('name'),
                                 backref='sets')
    
    def append(self, a):
        self.keys[a.name] = a

    def remove(self, item):
        del self.keys[a.name]

    def __repr__(self):
        return "Annotation Key Set (" + str(self.name) + ")"

    def __iter__(self):
        """Default iterator is over self.annotation_keys"""
        for key in self.keys.itervalues():
            yield key

    def __len__(self):
        return len(self.keys)


class AnnotationKey(SnorkelBase):
    """
    The Annotation key table is a mapping from unique string names to integer id numbers.
    These strings uniquely identify who or what produced an annotation.
    """
    __tablename__ = 'annotation_key'
    id        = Column(Integer, primary_key=True)
    name      = Column(String, unique=True, nullable=False)

    def __repr__(self):
        return str(self.__class__.__name__) + " (" + str(self.name) + ")"


# TODO: Make this whole thing polymorphic instead now that only one AnnotationKey class?
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

    # The key is the "name" or "type" of the Annotation- e.g. the name of a feature, or of a human annotator
    @declared_attr
    def key_id(cls):
        return Column('annotation_key_id', Integer, ForeignKey('annotation_key.id'), nullable=False)

    @declared_attr
    def key(cls):
        return relationship('AnnotationKey', backref=backref(camel_to_under(cls.__name__) + 's', cascade='all'))

    # Every annotation is with respect to a candidate
    @declared_attr
    def candidate_id(cls):
        return Column('candidate_id', Integer, ForeignKey('candidate.id'), nullable=False)

    @declared_attr
    def candidate(cls):
        return relationship('Candidate', backref=backref(camel_to_under(cls.__name__) + 's', \
                    cascade_backrefs=False))

    # NOTE: What remains to be defined in the subclass is the **value**

    # Table arguments
    @declared_attr
    def __table_args__(cls):
        return (
            UniqueConstraint('annotation_key_id',
                             'candidate_id',
                             name=camel_to_under(cls.__name__) + '_unique_key_candidate_pair'),
        )

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.key.name) + " = " + str(self.value) + ")"


class Label(AnnotationMixin, SnorkelBase):
    """
    A discrete label associated with a Candidate, indicating a target prediction value.

    Labels are used to represent both human-provided annotations and the output of labeling functions.

    A Label's LabelKey identifies the person or labeling function that provided the Label.
    """
    value = Column(Integer, nullable=False)


class Feature(AnnotationMixin, SnorkelBase):
    """
    An element of a representation of a Candidate in a feature space.

    A Feature's FeatureKey identifies the definition of the Feature, e.g., a function that implements it
    or the library name and feature name in an automatic featurization library.
    """
    value = Column(Float, nullable=False)


class Prediction(AnnotationMixin, SnorkelBase):
    """
    A probability associated with a Candidate, indicating the degree of belief that the Candidate is true.

    A Prediction's PredictionKey indicates which process or method produced the Prediction, e.g., which
    model with which ParameterSet.
    """
    value = Column(Float, nullable=False)
