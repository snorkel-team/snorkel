from .meta import SnorkelBase, snorkel_postgres
from sqlalchemy import Column, String, Integer, Text, ForeignKey
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import relationship
from sqlalchemy.types import PickleType


class Context(SnorkelBase):
    """A piece of content."""
    __tablename__ = 'context'
    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False)

    candidates = relationship('Candidate', backref='context', cascade_backrefs=False)

    __mapper_args__ = {
        'polymorphic_identity': 'context',
        'polymorphic_on': type
    }


class Corpus(Context):
    """
    A Corpus holds a set of Documents.

    Default iterator is over (Document, Sentence) tuples.
    """
    __tablename__ = 'corpus'
    id = Column(Integer, ForeignKey('context.id'), nullable=False)
    name = Column(String, primary_key=True)

    __mapper_args__ = {
        'polymorphic_identity': 'corpus',
    }

    def __repr__(self):
        return "Corpus (" + str(self.name) + ")"

    def __iter__(self):
        """Default iterator is over (document, document.sentences) tuples"""
        for doc in self.documents:
            yield (doc, doc.sentences)

    def get_sentences(self):
        return [sentence for doc in self.documents for sentence in doc.sentences]


class Document(Context):
    """An object in a Corpus."""
    __tablename__ = 'document'
    id = Column(Integer, ForeignKey('context.id'), nullable=False)
    name = Column(String, primary_key=True)
    corpus_id = Column(Integer, ForeignKey('corpus.id'), primary_key=True)
    corpus = relationship('Corpus', backref='documents', foreign_keys=corpus_id)
    file = Column(String)
    attribs = Column(PickleType)

    __mapper_args__ = {
        'polymorphic_identity': 'document',
    }

    def __repr__(self):
        return "Document" + str((self.name, self.corpus))


class Sentence(Context):
    """A sentence Context in a Document."""
    __tablename__ = 'sentence'
    id = Column(Integer, ForeignKey('context.id'))
    document_id = Column(Integer, ForeignKey('document.id'), primary_key=True)
    document = relationship('Document', backref='sentences', foreign_keys=document_id)
    position = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    if snorkel_postgres:
        words = Column(postgresql.ARRAY(String), nullable=False)
        char_offsets = Column(postgresql.ARRAY(Integer), nullable=False)
        lemmas = Column(postgresql.ARRAY(String))
        poses = Column(postgresql.ARRAY(String))
        dep_parents = Column(postgresql.ARRAY(Integer))
        dep_labels = Column(postgresql.ARRAY(String))
    else:
        words = Column(PickleType, nullable=False)
        char_offsets = Column(PickleType, nullable=False)
        lemmas = Column(PickleType)
        poses = Column(PickleType)
        dep_parents = Column(PickleType)
        dep_labels = Column(PickleType)

    __mapper_args__ = {
        'polymorphic_identity': 'sentence',
    }

    def __repr__(self):
        return "Sentence" + str((self.document, self.position, self.text))
