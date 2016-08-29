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

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)


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

    def __eq__(self, other):
        try:
            return self.annotator == other.annotator and self.candidate == other.candidate
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.annotator != other.annotator or self.candidate != other.candidate
        except AttributeError:
            return True

    def __hash__(self):
        return hash(self.annotator) + hash(self.candidate)
