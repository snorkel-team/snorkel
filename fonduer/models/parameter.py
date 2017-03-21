from .meta import SnorkelBase
from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship, backref


class ParameterSet(SnorkelBase):
    __tablename__ = 'parameter_set'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    def __repr__(self):
        return "Parameter Set (%s)" % self.name

    def __iter__(self):
        """Default iterator is over Parameters."""
        for parameter in self.parameters:
            yield parameter


class Parameter(SnorkelBase):
    __tablename__ = 'parameter'

    feature_key_id = Column(Integer, ForeignKey('annotation_key.id'), primary_key=True)
    feature_key = relationship('AnnotationKey', backref=backref('parameters', cascade_backrefs=False), cascade_backrefs=False)
    set_id = Column(Integer, ForeignKey('parameter_set.id'), primary_key=True)
    set = relationship('ParameterSet', backref=backref('parameters', cascade='all, delete-orphan'))
    value = Column(Float, nullable=False)

    def __repr__(self):
        return "Parameter (%s, %s)" % (self.feature_key.name, self.value)
