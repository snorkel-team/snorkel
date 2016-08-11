from .meta import SnorkelBase
from sqlalchemy import Column, String, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship, backref


class Annotator(SnorkelBase):
    """A human who provides annotations."""
    __tablename__ = 'annotator'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    def __repr__(self):
        return "Annotator (" + str(self.name) + ")"


class Annotation(SnorkelBase):
    """An annotation of a Candidate.

    Indicates whether the Candidate is true, false, or unknown. Add an Annotator to record who
    provided the annotation."""
    __tablename__ = 'annotation'
    id = Column(Integer, primary_key=True)
    annotator_id = Column(Integer, ForeignKey('annotator.id'))
    annotator = relationship('Annotator', backref=backref('annotations', cascade='all'))
    candidate_id = Column(Integer, ForeignKey('candidate.id'))
    candidate = relationship('Candidate', backref=backref('annotations', cascade_backrefs=False))
    value = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint(annotator_id, candidate_id),
    )

    def __repr__(self):
        return "Annotation (" + str(self.value) + " by " + str(self.annotator.name) + ")"
