from .meta import SnorkelSession, SnorkelBase
from .context import Context
from sqlalchemy import Table, Column, String, Integer, ForeignKey, ForeignKeyConstraint, UniqueConstraint
from sqlalchemy.orm import relationship, backref


class CandidateSet(SnorkelBase):
    """A named collection of Candidate objects."""
    __tablename__ = 'candidate_set'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    def append(self, item):
        self.candidates.append(item)

    def remove(self, item):
        self.candidates.remove(item)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)

    def __iter__(self):
        """Default iterator is over self.candidates"""
        for candidate in self.candidates:
            yield candidate

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, key):
        return self.candidates[key]

    def stats(self, gold_set=None):
        """Print diagnostic stats about CandidateSet."""
        session = SnorkelSession.object_session(self)

        # NOTE: This is the number of contexts that had non-zero number of candidates!
        nc = session.query(Context.id).join(Candidate).filter(CandidateSet.name == self.name).distinct().count()
        print "=" * 80
        print "%s candidates in %s contexts" % (self.__len__(), nc)
        print "Avg. # of candidates / context*: %0.1f" % (self.__len__() / float(nc),)
        if gold_set is not None:
            print "-" * 80
            print "Overlaps with %0.2f%% of gold set" % (len(gold_set.intersection(self)) / float(len(gold_set)),)
        print "=" * 80
        print "*Only counting contexts with non-zero number of candidates."

    def __repr__(self):
        return "Candidate Set (" + str(self.name) + ")"


class Candidate(SnorkelBase):
    """
    An abstract candidate relation.
    """
    __tablename__ = 'candidate'
    id = Column(Integer, primary_key=True)
    candidate_set_id = Column(Integer, ForeignKey('candidate_set.id'))
    set = relationship('CandidateSet', backref=backref('candidates', cascade='all, delete-orphan'))
    type = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint(id, candidate_set_id),
    )

    __mapper_args__ = {
        'polymorphic_identity': 'candidate',
        'polymorphic_on': type
    }


class SpanPair(Candidate):
    """
    A pair of Span Candidates, representing a relation from Span 0 to Span 1.
    """
    __table__ = Table('span_pair', SnorkelBase.metadata,
                      Column('id', Integer, primary_key=True),
                      Column('candidate_set_id', Integer),
                      Column('span0_id', Integer),
                      Column('span1_id', Integer),
                      ForeignKeyConstraint(['id', 'candidate_set_id'], ['candidate.id', 'candidate.candidate_set_id']),
                      ForeignKeyConstraint(['span0_id'], ['span.id']),
                      ForeignKeyConstraint(['span1_id'], ['span.id'])
                      )

    __table_args__ = (
        UniqueConstraint(__table__.c.candidate_set_id, __table__.c.span0_id, __table__.c.span1_id),
    )

    span0 = relationship('Span', backref=backref('span_source_pairs', cascade_backrefs=False),
                          cascade_backrefs=False, foreign_keys='SpanPair.span0_id')
    span1 = relationship('Span', backref=backref('span_dest_pairs', cascade_backrefs=False),
                          cascade_backrefs=False, foreign_keys='SpanPair.span1_id')

    __mapper_args__ = {
        'polymorphic_identity': 'span_pair',
    }

    def __eq__(self, other):
        try:
            return self.span0 == other.span0 and self.span1 == other.span1
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.span0 != other.span0 or self.span1 != other.span1
        except AttributeError:
            return True

    def __hash__(self):
        return hash(self.span0) + hash(self.span1)

    def __getitem__(self, key):
        if key == 0:
            return self.span0
        elif key == 1:
            return self.span1
        else:
            raise KeyError('Valid keys are 0 and 1.')

    def __repr__(self):
        return "SpanPair(%s, %s)" % (self.span0, self.span1)
