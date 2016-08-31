from .meta import SnorkelSession, SnorkelBase
from .context import Context
from sqlalchemy import Table, Column, String, Integer, ForeignKey
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import relationship
from snorkel.models import snorkel_engine

candidate_set_candidate_association = Table('candidate_set_candidate_association', SnorkelBase.metadata,
                                            Column('candidate_set_id', Integer, ForeignKey('candidate_set.id')),
                                            Column('candidate_id', Integer, ForeignKey('candidate.id')))


class CandidateSet(SnorkelBase):
    """
    A set of Candidates, uniquely identified by a name.

    CandidateSets have many-to-many relationships with Candidates, so users can create new
    subsets, supersets, etc.
    """
    __tablename__ = 'candidate_set'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    def append(self, item):
        self.candidates.append(item)

    def remove(self, item):
        self.candidates.remove(item)

    def __repr__(self):
        return "Candidate Set (" + str(self.name) + ")"

    def __iter__(self):
        """Default iterator is over self.candidates"""
        for candidate in self.candidates:
            yield candidate

    def __len__(self):
        return len(self.candidates)

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


class Candidate(SnorkelBase):
    """
    An abstract candidate relation.

    New relation types can be defined by extending this class. For example,
    a new parent-child relation can be defined as

    >>> from snorkel.models.candidate import Candidate
    >>> from sqlalchemy import Column, String, ForeignKey
    >>> from sqlalchemy.orm import relationship
    >>>
    >>> class NewType(Candidate):
    >>>     # Declares name for storage table
    >>>     __tablename__ = 'newtype'
    >>>     # Connects NewType records to generic Candidate records
    >>>     id = Column(Integer, ForeignKey('candidate.id'))
    >>>
    >>>     # Polymorphism information for SQLAlchemy
    >>>     __mapper_args__ = {
    >>>         'polymorphic_identity': 'newtype',
    >>>     }
    >>>
    >>>     # Relation arguments declared as a compound primary key,
    >>>     # with each argument a context. More specific Context types can also be used.
    >>>     parent_id = Column(Integer, ForeignKey('context.id'), primary_key=True)
    >>>     parent = relationship('Context', foreign_keys=parent_id)
    >>>     child_id = Column(Integer, ForeignKey('context.id'), primary_key=True)
    >>>     child = relationship('Context', foreign_keys=child_id)
    >>>
    >>> # The entire storage schema, including NewType, can now be initialized with the following import
    >>> import snorkel.models
    """
    __tablename__ = 'candidate'
    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'candidate',
        'polymorphic_on': type
    }

    def get_arguments(self):
        #return [self.__getattribute__(key.name) for key in inspect(type(self)).primary_key]


    def __getitem__(self, key):
        return self.get_arguments()[key]

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join(map(str, self.get_arguments())))


def candidate_subclass(class_name, table_name, args):
    """
    Creates and returns a Candidate subclass with provided argument names, which are Context type.
    Creates the table in DB if does not exist yet.

    Similar in spirit to collections.namedtuple.
    """
    class_attribs = {

        # Declares name for storage table
        '__tablename__' : table_name,
                
        # Connects ChemicalDisease records to generic Candidate records
        'id' : Column(Integer, ForeignKey('candidate.id')),
                
        # Polymorphism information for SQLAlchemy
        '__mapper_args__' : {'polymorphic_identity': table_name},

        # Helper method to get argument names
        '__argnames__' : args
    }
        
    # Create named arguments
    for arg in args:
        class_attribs[arg + '_id'] = Column(Integer, ForeignKey('context.id'), primary_key=True)
        class_attribs[arg]         = relationship('Context', foreign_keys=class_attribs[arg + '_id'])
            
    # Create class
    C = type(class_name, (Candidate,), class_attribs)
        
    # Create table in DB
    if not snorkel_engine.dialect.has_table(snorkel_engine, table_name):
        C.__table__.create(bind=snorkel_engine)
    return C
