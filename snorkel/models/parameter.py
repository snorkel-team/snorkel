from .meta import SnorkelBase
from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship, backref

# Development note:
# Without a single AnnotationKey class (either by only having one such table, or via polymorphism), we
# would have to make a new Parameter class for each AnnotationKey type... for now, there doesn't seem to be much
# need to save the generative model parameters, so just making this reference FeatureKey.
# We can revisit this down the road based on which learning components remain internal to the core Snorkel codebase
class Parameter(SnorkelBase):
    __tablename__ = 'parameter'

    feature_key_id = Column(Integer, ForeignKey('feature_key.id', ondelete='CASCADE'), primary_key=True)
    feature_key    = relationship('FeatureKey', backref=backref('parameters', cascade='all, delete-orphan', cascade_backrefs=False), cascade_backrefs=False)
    value          = Column(Float, nullable=False)
    version        = Column(Integer, default=0)

    def __repr__(self):
        return "Parameter (%s, %s)" % (self.feature_key.name, self.value)
