from .meta import SnorkelSession, SnorkelBase
from .context import Context
from sqlalchemy import Table, Column, String, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship, backref
from snorkel.models import snorkel_engine
from snorkel.utils import camel_to_under


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
    candidates = relationship('Candidate', secondary=candidate_set_candidate_association, backref='sets', \
                    lazy='dynamic')

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

    def __getitem__(self, key):
        return self.candidates[key]

    def __len__(self):
        return self.candidates.count()

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

    New relation types should be defined by calling candidate_subclass(), **not** subclassing
    this class directly.
    """
    __tablename__ = 'candidate'
    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'candidate',
        'polymorphic_on': type
    }

    def get_arguments(self):
        return tuple(getattr(self, name) for name in self.__argnames__)

    def __getitem__(self, key):
        return self.get_arguments()[key]

    def __repr__(self):
        return u"%s(%s)" % (self.__class__.__name__, u", ".join(map(unicode, self.get_arguments())))


def candidate_subclass(class_name, args, table_name=None):
    """
    Creates and returns a Candidate subclass with provided argument names, which are Context type.
    Creates the table in DB if does not exist yet.

    Import using:

    .. code-block:: python

        from snorkel.models import candidate_subclass

    :param class_name: The name of the class, should be "camel case" e.g. NewCandidateClass
    :param args: A list of names of consituent arguments, which refer to the Contexts--representing mentions--that
        comprise the candidate
    :param table_name: The name of the corresponding table in DB; if not provided, is converted from camel case
        by default, e.g. new_candidate_class
    """
    table_name = camel_to_under(class_name) if table_name is None else table_name
    class_attribs = {

        # Declares name for storage table
        '__tablename__' : table_name,
                
        # Connects ChemicalDisease records to generic Candidate records
        'id' : Column(Integer, ForeignKey('candidate.id'), primary_key=True),
                
        # Polymorphism information for SQLAlchemy
        '__mapper_args__' : {'polymorphic_identity': table_name},

        # Helper method to get argument names
        '__argnames__' : args
    }
        
    # Create named arguments
    unique_con_args = []
    for arg in args:
        class_attribs[arg + '_id'] = Column(Integer, ForeignKey('context.id'))
        class_attribs[arg]         = relationship('Context',
                                                  backref=backref(table_name + '_' + arg + 's', cascade_backrefs=False),
                                                  cascade_backrefs=False,
                                                  foreign_keys=class_attribs[arg + '_id'])
        unique_con_args.append(class_attribs[arg + '_id'])

    class_attribs['__table_args__'] = (UniqueConstraint(*unique_con_args),)

    # Create class
    C = type(class_name, (Candidate,), class_attribs)
        
    # Create table in DB
    if not snorkel_engine.dialect.has_table(snorkel_engine, table_name):
        C.__table__.create(bind=snorkel_engine)
    return C
