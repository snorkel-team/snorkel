from .meta import SnorkelBase
from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship, backref

class Parameter(SnorkelBase):
    __tablename__ = 'parameter'

    feature_key_id = Column(Integer, ForeignKey('_key.id'), primary_key=True)
    feature_key = relationship('AnnotationKey', backref=backref('parameters', cascade='all, delete-orphan', cascade_backrefs=False), cascade_backrefs=False)
    value = Column(Float, nullable=False)

    def __repr__(self):
        return "Parameter (%s, %s)" % (self.feature_key.name, self.value)
