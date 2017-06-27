from sqlalchemy import (
    Column, String, Integer, Float, Boolean, ForeignKey, UniqueConstraint,
    MetaData
)
from sqlalchemy.orm import relationship, backref
from functools import partial

from .meta import SnorkelBase
from ..models import snorkel_engine
from ..utils import camel_to_under


class Candidate(SnorkelBase):
    """
    An abstract candidate relation.

    New relation types should be defined by calling candidate_subclass(),
    **not** subclassing this class directly.
    """
    __tablename__ = 'candidate'
    id          = Column(Integer, primary_key=True)
    type        = Column(String, nullable=False)
    split       = Column(Integer, nullable=False, default=0, index=True)

    __mapper_args__ = {
        'polymorphic_identity': 'candidate',
        'polymorphic_on': type
    }

    # __table_args__ = {"extend_existing" : True}

    def get_contexts(self):
        """Get a tuple of the consituent contexts making up this candidate"""
        return tuple(getattr(self, name) for name in self.__argnames__)

    def get_parent(self):
        # Fails if both contexts don't have same parent
        p = [c.get_parent() for c in self.get_contexts()]
        if p.count(p[0]) == len(p):
            return p[0]
        else:
            raise Exception("Contexts do not all have same parent")

    def get_cids(self):
        """Get a tuple of the canonical IDs (CIDs) of the contexts making up 
        this candidate"""
        return tuple(getattr(self, name + "_cid") for name in self.__argnames__)

    def __len__(self):
        return len(self.__argnames__)

    def __getitem__(self, key):
        return self.get_contexts()[key]

    def __repr__(self):
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(map(str, self.get_contexts()))
        )


def candidate_subclass(class_name, args, table_name=None, cardinality=None,
    values=None):
    """
    Creates and returns a Candidate subclass with provided argument names, 
    which are Context type. Creates the table in DB if does not exist yet.

    Import using:

    .. code-block:: python

        from snorkel.models import candidate_subclass

    :param class_name: The name of the class, should be "camel case" e.g. 
        NewCandidate
    :param args: A list of names of consituent arguments, which refer to the 
        Contexts--representing mentions--that comprise the candidate
    :param table_name: The name of the corresponding table in DB; if not 
        provided, is converted from camel case by default, e.g. new_candidate
    :param cardinality: The cardinality of the variable corresponding to the
        Candidate. By default is 2 i.e. is a binary value, e.g. is or is not
        a true mention.
    """
    if table_name is None:
        table_name = camel_to_under(class_name)

    # If cardinality and values are None, default to binary classification
    if cardinality is None and values is None:
        values = [True, False]
        cardinality = 2
    
    # Else use values if present, and validate proper input
    elif values is not None:
        if cardinality is not None and len(values) != cardinality:
            raise ValueError("Number of values must match cardinality.")
        if None in values:
            raise ValueError("`None` is a protected value.")
        if any([type(v) == int for v in values]):
            raise ValueError("Values cannot be integers.")
        cardinality = len(values)

    # If cardinality is specified but not values, fill in with ints
    elif cardinality is not None:
        values = range(cardinality)

    # Set the class attributes == the columns in the database
    class_attribs = {

        # Declares name for storage table
        '__tablename__' : table_name,
                
        # Connects candidate_subclass records to generic Candidate records
        'id' : Column(
            Integer,
            ForeignKey('candidate.id', ondelete='CASCADE'),
            primary_key=True
        ),

        # Store values & cardinality information in the class only
        'values' : values,
        'cardinality' : cardinality,
                
        # Polymorphism information for SQLAlchemy
        '__mapper_args__' : {'polymorphic_identity': table_name},

        # Helper method to get argument names
        '__argnames__' : args,
    }
        
    # Create named arguments, i.e. the entity mentions comprising the relation 
    # mention
    # For each entity mention: id, cid ("canonical id"), and pointer to Context
    unique_args = []
    for arg in args:

        # Primary arguments are constituent Contexts, and their ids
        class_attribs[arg + '_id'] = Column(
            Integer, ForeignKey('context.id', ondelete='CASCADE'))
        class_attribs[arg] = relationship(
            'Context',
            backref=backref(
                table_name + '_' + arg + 's',
                cascade_backrefs=False,
                cascade='all, delete-orphan'
            ),
            cascade_backrefs=False,
            foreign_keys=class_attribs[arg + '_id']
        )
        unique_args.append(class_attribs[arg + '_id'])

        # Canonical ids, to be set post-entity normalization stage
        class_attribs[arg + '_cid'] = Column(String)

    # Add unique constraints to the arguments
    class_attribs['__table_args__'] = (
        UniqueConstraint(*unique_args),
        # Note: This still doesn't fix issue...
        {'keep_existing' : True}
    )

    # Create class
    C = type(class_name, (Candidate,), class_attribs)
        
    # Create table in DB
    if not snorkel_engine.dialect.has_table(snorkel_engine, table_name):
        C.__table__.create(bind=snorkel_engine)
    return C


class Marginal(SnorkelBase):
    """
    A marginal probability corresponding to a (Candidate, value) pair.

    Represents:

        P(candidate = value) = probability

    @training: If True, this is a training marginal; otherwise is end prediction
    """
    __tablename__ = 'marginal'
    id           = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, 
                        ForeignKey('candidate.id', ondelete='CASCADE'))
    training     = Column(Boolean, default=True)
    value        = Column(Integer, nullable=False, default=1)
    probability  = Column(Float, nullable=False, default=0.0)
    
    __table_args__ = (
        UniqueConstraint(candidate_id, training, value),
    )

    def __repr__(self):
        label = "Training" if self.training else "Predicted"
        return "<%s Marginal: P(%s == %s) = %s>" % \
            (label, self.candidate_id, self.value, self.probability)
