from .meta import SnorkelBase, snorkel_postgres
from sqlalchemy import Column, String, Integer, Text, ForeignKey, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import relationship, backref
from sqlalchemy.types import PickleType
from sqlalchemy.sql import select, text


class Context(SnorkelBase):
    """
    A piece of content from which Candidates are composed.
    """
    __tablename__ = 'context'
    id            = Column(Integer, primary_key=True)
    type          = Column(String, nullable=False)
    stable_id     = Column(String, unique=True, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'context',
        'polymorphic_on': type
    }

    def get_parent(self):
        raise NotImplementedError()

    def get_children(self):
        raise NotImplementedError()

    def get_sentence_generator(self):
        raise NotImplementedError()


class Document(Context):
    """
    A root Context.
    """
    __tablename__ = 'document'
    id = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)
    name = Column(String, unique=True, nullable=False)
    meta = Column(PickleType)

    __mapper_args__ = {
        'polymorphic_identity': 'document',
    }

    def get_parent(self):
        return None

    def get_children(self):
        return self.sentences

    def get_sentence_generator(self):
        for sentence in self.sentences:
            yield sentence

    def __repr__(self):
        return "Document " + str(self.name)


class Sentence(Context):
    """A sentence Context in a Document."""
    __tablename__ = 'sentence'
    id = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)
    document_id = Column(Integer, ForeignKey('document.id', ondelete='CASCADE'))
    position = Column(Integer, nullable=False)
    document = relationship('Document', backref=backref('sentences', order_by=position, cascade='all, delete-orphan'), foreign_keys=document_id)
    text = Column(Text, nullable=False)
    if snorkel_postgres:
        words             = Column(postgresql.ARRAY(String), nullable=False)
        char_offsets      = Column(postgresql.ARRAY(Integer), nullable=False)
        abs_char_offsets  = Column(postgresql.ARRAY(Integer), nullable=False)
        lemmas            = Column(postgresql.ARRAY(String))
        pos_tags          = Column(postgresql.ARRAY(String))
        ner_tags          = Column(postgresql.ARRAY(String))
        dep_parents       = Column(postgresql.ARRAY(Integer))
        dep_labels        = Column(postgresql.ARRAY(String))
        entity_cids       = Column(postgresql.ARRAY(String))
        entity_types      = Column(postgresql.ARRAY(String))
    else:
        words             = Column(PickleType, nullable=False)
        char_offsets      = Column(PickleType, nullable=False)
        abs_char_offsets  = Column(PickleType, nullable=False)
        lemmas            = Column(PickleType)
        pos_tags          = Column(PickleType)
        ner_tags          = Column(PickleType)
        dep_parents       = Column(PickleType)
        dep_labels        = Column(PickleType)
        entity_cids       = Column(PickleType)
        entity_types      = Column(PickleType)

    __mapper_args__ = {
        'polymorphic_identity': 'sentence',
    }

    __table_args__ = (
        UniqueConstraint(document_id, position),
    )

    def get_parent(self):
        return self.document

    def get_children(self):
        return self.spans

    def _asdict(self):
        return {
            'id': self.id,
            'document': self.document,
            'position': self.position,
            'text': self.text,
            'words': self.words,
            'char_offsets': self.char_offsets,
            'lemmas': self.lemmas,
            'pos_tags': self.pos_tags,
            'ner_tags': self.ner_tags,
            'dep_parents': self.dep_parents,
            'dep_labels': self.dep_labels,
            'entity_cids': self.entity_cids,
            'entity_types': self.entity_types
        }

    def get_sentence_generator(self):
        yield self

    def __repr__(self):
        return "Sentence(%s,%s,%s)" % (self.document, self.position, self.text.encode('utf-8'))


class TemporaryContext(object):
    """
    A context which does not incur the overhead of a proper ORM-based Context object.
    The TemporaryContext class is specifically for the candidate extraction process, during which a CandidateSpace
    object will generate many TemporaryContexts, which will then be filtered by Matchers prior to materialization
    of Candidates and constituent Context objects.

    Every Context object has a corresponding TemporaryContext object from which it inherits.

    A TemporaryContext must have specified equality / set membership semantics, a stable_id for checking
    uniqueness against the database, and a promote() method which returns a corresponding Context object.
    """
    def __init__(self):
        self.id = None

    def load_id_or_insert(self, session):
        if self.id is None:
            stable_id = self.get_stable_id()
            id = session.execute(select([Context.id]).where(Context.stable_id == stable_id)).first()
            if id is None:
                self.id = session.execute(
                        Context.__table__.insert(),
                        {'type': self._get_table_name(), 'stable_id': stable_id}).inserted_primary_key[0]
                insert_args = self._get_insert_args()
                insert_args['id'] = self.id
                session.execute(text(self._get_insert_query()), insert_args)
            else:
                self.id = id[0]

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()

    def _get_polymorphic_identity(self):
        raise NotImplementedError()

    def get_stable_id(self):
        raise NotImplementedError()

    def _get_table_name(self):
        raise NotImplementedError()

    def _get_insert_query(self):
        raise NotImplementedError()

    def _get_insert_args(self):
        raise NotImplementedError()


class TemporarySpan(TemporaryContext):
    """The TemporaryContext version of Span"""
    def __init__(self, sentence, char_start, char_end, meta=None):
        super(TemporarySpan, self).__init__()
        self.sentence     = sentence  # The sentence Context of the Span
        self.char_end   = char_end
        self.char_start = char_start
        self.meta       = meta

    def __len__(self):
        return self.char_end - self.char_start + 1

    def __eq__(self, other):
        try:
            return self.sentence == other.sentence and self.char_start == other.char_start \
                and self.char_end == other.char_end
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.sentence != other.sentence or self.char_start != other.char_start \
                or self.char_end != other.char_end
        except AttributeError:
            return True

    def __hash__(self):
        return hash(self.sentence) + hash(self.char_start) + hash(self.char_end)

    def get_stable_id(self):
        return construct_stable_id(self.sentence, self._get_polymorphic_identity(), self.char_start, self.char_end)

    def _get_table_name(self):
        return 'span'

    def _get_polymorphic_identity(self):
        return 'span'

    def _get_insert_query(self):
        return """INSERT INTO span VALUES(:id, :sentence_id, :char_start, :char_end, :meta)"""

    def _get_insert_args(self):
        return {'sentence_id' : self.sentence.id,
                'char_start': self.char_start,
                'char_end'  : self.char_end,
                'meta'      : self.meta}

    def get_word_start(self):
        return self.char_to_word_index(self.char_start)

    def get_word_end(self):
        return self.char_to_word_index(self.char_end)

    def get_n(self):
        return self.get_word_end() - self.get_word_start() + 1

    def char_to_word_index(self, ci):
        """Given a character-level index (offset), return the index of the **word this char is in**"""
        i = None
        for i, co in enumerate(self.sentence.char_offsets):
            if ci == co:
                return i
            elif ci < co:
                return i-1
        return i

    def word_to_char_index(self, wi):
        """Given a word-level index, return the character-level index (offset) of the word's start"""
        return self.sentence.char_offsets[wi]

    def get_attrib_tokens(self, a='words'):
        """Get the tokens of sentence attribute _a_ over the range defined by word_offset, n"""
        return self.sentence.__getattribute__(a)[self.get_word_start():self.get_word_end() + 1]

    def get_attrib_span(self, a, sep=" "):
        """Get the span of sentence attribute _a_ over the range defined by word_offset, n"""
        # NOTE: Special behavior for words currently (due to correspondence with char_offsets)
        if a == 'words':
            return self.sentence.text[self.char_start:self.char_end + 1]
        else:
            return sep.join(self.get_attrib_tokens(a))

    def get_span(self, sep=" "):
        return self.get_attrib_span('words', sep)

    def __contains__(self, other_span):
        return other_span.char_start >= self.char_start and other_span.char_end <= self.char_end

    def __getitem__(self, key):
        """
        Slice operation returns a new candidate sliced according to **char index**
        Note that the slicing is w.r.t. the candidate range (not the abs. sentence char indexing)
        """
        if isinstance(key, slice):
            char_start = self.char_start if key.start is None else self.char_start + key.start
            if key.stop is None:
                char_end = self.char_end
            elif key.stop >= 0:
                char_end = self.char_start + key.stop - 1
            else:
                char_end = self.char_end + key.stop
            return self._get_instance(char_start=char_start, char_end=char_end, sentence=self.sentence)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return '%s("%s", sentence=%s, chars=[%s,%s], words=[%s,%s])' \
            % (self.__class__.__name__, self.get_span().encode('utf-8'), self.sentence.id, self.char_start, self.char_end,
               self.get_word_start(), self.get_word_end())

    def _get_instance(self, **kwargs):
        return TemporarySpan(**kwargs)


class Span(Context, TemporarySpan):
    """
    A span of characters, identified by Context id and character-index start, end (inclusive).

    char_offsets are **relative to the Context start**
    """
    __tablename__ = 'span'
    id = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)
    sentence_id = Column(Integer, ForeignKey('sentence.id', ondelete='CASCADE'))
    char_start = Column(Integer, nullable=False)
    char_end = Column(Integer, nullable=False)
    meta = Column(PickleType)

    __table_args__ = (
        UniqueConstraint(sentence_id, char_start, char_end),
    )

    __mapper_args__ = {
        'polymorphic_identity': 'span',
        'inherit_condition': (id == Context.id)
    }

    sentence = relationship('Sentence', backref=backref('spans', cascade='all, delete-orphan'), order_by=char_start, foreign_keys=sentence_id)

    def get_parent(self):
        return self.sentence

    def get_children(self):
        return None

    def _get_instance(self, **kwargs):
        return Span(**kwargs)

    # We redefine these to use default semantics, overriding the operators inherited from TemporarySpan
    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)


def split_stable_id(stable_id):
    """
    Split stable id, returning:
        * Document (root) stable ID
        * Context polymorphic type
        * Character offset start, end *relative to document start*
    Returns tuple of four values.
    """
    split1 = stable_id.split('::')
    if len(split1) == 2:
        split2 = split1[1].split(':')
        if len(split2) == 3:
            return split1[0], split2[0], int(split2[1]), int(split2[2])
    raise ValueError("Malformed stable_id:", stable_id)


def construct_stable_id(parent_context, polymorphic_type, relative_char_offset_start, relative_char_offset_end):
    """Contruct a stable ID for a Context given its parent and its character offsets relative to the parent"""
    doc_id, _, parent_doc_char_start, _ = split_stable_id(parent_context.stable_id)
    start = parent_doc_char_start + relative_char_offset_start
    end   = parent_doc_char_start + relative_char_offset_end
    return "%s::%s:%s:%s" % (doc_id, polymorphic_type, start, end)
