from .meta import SnorkelBase, snorkel_postgres
from sqlalchemy import Column, String, Integer, Text, ForeignKey
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import relationship
from sqlalchemy.types import PickleType


class Corpus(SnorkelBase):
    """
    A Corpus holds a set of Documents and associated Contexts.

    Default iterator is over (Document, Context) tuples.
    """

    __tablename__ = 'corpus'
    id = Column(String, primary_key=True)

    documents = relationship('Document', backref='corpus')

    def __repr__(self):
        return "Corpus" + str((self.id,))

    def __iter__(self):
        """Default iterator is over (document, document.contexts) tuples"""
        for doc in self.documents:
            yield (doc, doc.contexts)

    def get_contexts(self):
        return [context for doc in self.documents for context in doc.contexts]


class Document(SnorkelBase):
    """An object in a Corpus."""
    __tablename__ = 'document'
    id = Column(String, primary_key=True)
    corpus_id = Column(String, ForeignKey('corpus.id'))
    file = Column(String)
    attribs = Column(PickleType)

    contexts = relationship('Context', backref='document')

    def __repr__(self):
        return "Document" + str((self.id, self.corpus_id, self.file, self.attribs))


class Context(SnorkelBase):
    """A piece of content contained in a Document."""
    __tablename__ = 'context'
    id = Column(String, primary_key=True)
    type = Column(String)
    document_id = Column(String, ForeignKey('document.id'))

    candidates = relationship('Candidate', backref='context', cascade_backrefs=False)

    __mapper_args__ = {
        'polymorphic_identity': 'context',
        'polymorphic_on': type
    }


class Sentence(Context):
    """A sentence Context in a Document."""
    __tablename__ = 'sentence'
    id = Column(String, ForeignKey('context.id'), primary_key=True)
    position = Column(Integer)
    text = Column(Text)
    if snorkel_postgres:
        words = Column(postgresql.ARRAY(String))
        lemmas = Column(postgresql.ARRAY(String))
        poses = Column(postgresql.ARRAY(String))
        dep_parents = Column(postgresql.ARRAY(Integer))
        dep_labels = Column(postgresql.ARRAY(String))
        char_offsets = Column(postgresql.ARRAY(Integer))
    else:
        words = Column(PickleType)
        lemmas = Column(PickleType)
        poses = Column(PickleType)
        dep_parents = Column(PickleType)
        dep_labels = Column(PickleType)
        char_offsets = Column(PickleType)

    __mapper_args__ = {
        'polymorphic_identity': 'sentence',
    }

    def __repr__(self):
        return "Sentence" + str((self.id, self.document_id, self.position, self.text, self.words, self.lemmas,
                                 self.poses, self.dep_parents, self.dep_labels, self.char_offsets))
