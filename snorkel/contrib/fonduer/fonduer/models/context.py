import pickle

from sqlalchemy import Column, String, Integer, Text, ForeignKey, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import select, text
from sqlalchemy.types import PickleType

from .meta import SnorkelBase, snorkel_postgres

INT_ARRAY_TYPE = postgresql.ARRAY(Integer) if snorkel_postgres else PickleType
STR_ARRAY_TYPE = postgresql.ARRAY(String)  if snorkel_postgres else PickleType


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
    id            = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)
    name          = Column(String, unique=True, nullable=False)
    text          = Column(String)
    meta          = Column(PickleType)

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
        return "Document " + str(self.name.encode('utf-8'))

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()

class Webpage(Document):
    """
    Declares name for storage table.
    """
    __tablename__ = 'webpage'
    id            = Column(Integer, ForeignKey('document.id', ondelete='CASCADE'), primary_key=True)
    # Connects NewType records to generic Context records
    url           = Column(String)
    host          = Column(String)
    page_type     = Column(String)
    raw_content   = Column(String)
    crawltime     = Column(String)
    all           = Column(String)

    # Polymorphism information for SQLAlchemy
    __mapper_args__ = {
        'polymorphic_identity': 'webpage',
    }

    # Rest of class definition here
    def __repr__(self):
        return "Webpage(id: %s..., url: %s...)" % (self.name[:10].encode('utf-8'), self.url[8:23].encode('utf-8'))

class Sentence(Context):
    """A sentence Context in a Document."""
    __tablename__ = 'sentence'
    id = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)
    document_id = Column(Integer, ForeignKey('document.id', ondelete='CASCADE'))
    position = Column(Integer, nullable=False)
    document = relationship('Document', backref=backref('sentences', order_by=position, cascade='all, delete-orphan'), foreign_keys=document_id)
    text = Column(Text, nullable=False)
    if snorkel_postgres:
        words        = Column(postgresql.ARRAY(String), nullable=False)
        char_offsets = Column(postgresql.ARRAY(Integer), nullable=False)
        lemmas       = Column(postgresql.ARRAY(String))
        pos_tags     = Column(postgresql.ARRAY(String))
        ner_tags     = Column(postgresql.ARRAY(String))
        dep_parents  = Column(postgresql.ARRAY(Integer))
        dep_labels   = Column(postgresql.ARRAY(String))
        entity_cids  = Column(postgresql.ARRAY(String))
        entity_types = Column(postgresql.ARRAY(String))
    else:
        words        = Column(PickleType, nullable=False)
        char_offsets = Column(PickleType, nullable=False)
        lemmas       = Column(PickleType)
        pos_tags     = Column(PickleType)
        ner_tags     = Column(PickleType)
        dep_parents  = Column(PickleType)
        dep_labels   = Column(PickleType)
        entity_cids  = Column(PickleType)
        entity_types = Column(PickleType)

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


class Table(Context):
    """A table Context in a Document."""
    __tablename__ = 'table'
    id            = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)
    document_id   = Column(Integer, ForeignKey('document.id'))
    position      = Column(Integer, nullable=False)
    document      = relationship('Document', backref=backref('tables', order_by=position, cascade='all, delete-orphan'), foreign_keys=document_id)

    __mapper_args__ = {
        'polymorphic_identity': 'table',
    }

    __table_args__ = (
        UniqueConstraint(document_id, position),
    )

    def __repr__(self):
        return "Table(Doc: %s, Position: %s)" % (self.document.name.encode('utf-8'), self.position)

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()

class Cell(Context):
    """A cell Context in a Document."""
    __tablename__ = 'cell'
    id            = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)
    document_id   = Column(Integer, ForeignKey('document.id'))
    table_id      = Column(Integer, ForeignKey('table.id'))
    position      = Column(Integer, nullable=False)
    document      = relationship('Document', backref=backref('cells', order_by=position, cascade='all, delete-orphan'), foreign_keys=document_id)
    table         = relationship('Table', backref = backref('cells', order_by=position, cascade='all, delete-orphan'), foreign_keys = table_id)
    row_start = Column(Integer)
    row_end = Column(Integer)
    col_start = Column(Integer)
    col_end = Column(Integer)
    html_tag = Column(Text)
    if snorkel_postgres:
        html_attrs = Column(postgresql.ARRAY(String))
        # html_anc_tags = Column(postgresql.ARRAY(String))
        # html_anc_attrs = Column(postgresql.ARRAY(String))
    else:
        html_attrs = Column(PickleType)
        # html_anc_tags = Column(PickleType)
        # html_anc_attrs = Column(PickleType)

    __mapper_args__ = {
        'polymorphic_identity': 'cell',
    }

    __table_args__ = (
        UniqueConstraint(document_id, table_id, position),
    )

    def __repr__(self):
        return ("Cell(Doc: %s, Table: %s, Row: %s, Col: %s, Pos: %s)" %
                (self.document.name.encode('utf-8'),
                 self.table.position,
                 tuple(set([self.row_start, self.row_end])),
                 tuple(set([self.col_start, self.col_end])),
                 self.position))

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()

class PhraseMixin(object):
    """A phrase Context in a Document."""
    def is_lingual(self):
        return False

    def is_visual(self):
        return False

    def is_tabular(self):
        return False

    def is_structural(self):
        return False

    def __repr__(self):
        return ("Phrase (Doc: %s, Index: %s, Text: %s)" %
                (self.document.name.encode('utf-8'),
                 self.phrase_idx,
                 self.text.encode('utf-8')))

class LingualMixin(object):
    """A collection of lingual attributes."""
    lemmas      = Column(STR_ARRAY_TYPE)
    pos_tags    = Column(STR_ARRAY_TYPE)
    ner_tags    = Column(STR_ARRAY_TYPE)
    dep_parents = Column(INT_ARRAY_TYPE)
    dep_labels  = Column(STR_ARRAY_TYPE)

    def is_lingual(self):
        return self.lemmas is not None

    def __repr__(self):
        return ("LingualPhrase (Doc: %s, Index: %s, Text: %s)" %
                (self.document.name.encode('utf-8'),
                 self.phrase_idx,
                 self.text.encode('utf-8')))

class TabularMixin(object):
    """A collection of tabular attributes."""
    @declared_attr
    def table_id(cls):
        return Column('table_id', ForeignKey('table.id'))

    @declared_attr
    def table(cls):
        return relationship('Table', backref=backref('phrases', cascade='all, delete-orphan'), foreign_keys=lambda: cls.table_id)

    @declared_attr
    def cell_id(cls):
        return Column('cell_id', ForeignKey('cell.id'))

    @declared_attr
    def cell(cls):
        return relationship('Cell', backref=backref('phrases', cascade='all, delete-orphan'), foreign_keys=lambda: cls.cell_id)

    row_start = Column(Integer)
    row_end   = Column(Integer)
    col_start = Column(Integer)
    col_end   = Column(Integer)
    position  = Column(Integer)

    def is_tabular(self):
        return self.table is not None

    def is_cellular(self):
        return self.cell is not None

    def __repr__(self):
        rows = tuple([self.row_start, self.row_end]) if self.row_start != self.row_end else self.row_start
        cols = tuple([self.col_start, self.col_end]) if self.col_start != self.col_end else self.col_start
        return ("TabularPhrase (Doc: %s, Table: %s, Row: %s, Col: %s, Index: %s, Text: %s)" %
            (self.document.name.encode('utf-8'),
            (lambda: cls.table).position,
            rows,
            cols,
            self.phrase_idx,
            self.text.encode('utf-8')))


class VisualMixin(object):
    """ A collection of visual attributes."""
    page   = Column(INT_ARRAY_TYPE)
    top    = Column(INT_ARRAY_TYPE)
    left   = Column(INT_ARRAY_TYPE)
    bottom = Column(INT_ARRAY_TYPE)
    right  = Column(INT_ARRAY_TYPE)

    def is_visual(self):
        return self.page is not None and self.page[0] is not None

    def __repr__(self):
        return ("VisualPhrase (Doc: %s, Page: %s, (T,B,L,R): (%d,%d,%d,%d), Text: %s)" %
            (self.document.name.encode('utf-8'),
            self.page, self.top, self.bottom, self.left, self.right,
            self.text.encode('utf-8')))


class StructuralMixin(object):
    """ A collection of structural attributes."""
    xpath      = Column(String)
    html_tag   = Column(String)
    html_attrs = Column(STR_ARRAY_TYPE)

    def is_structural(self):
        return self.html_tag is not None

    def __repr__(self):
        return ("StructuralPhrase (Doc: %s, Tag: %s, Text: %s)" %
            (self.document.name.encode('utf-8'),
            self.html_tag,
            self.text.encode('utf-8')))


# PhraseMixin must come last in arguments to not ovewrite is_* methods
# class Phrase(Context, StructuralMixin, PhraseMixin): # Memex variant
class Phrase(Context, TabularMixin, LingualMixin, VisualMixin, StructuralMixin, PhraseMixin):
    """A Phrase subclass with Lingual, Tabular, Visual, and HTML attributes."""
    __tablename__ = 'phrase'
    id            = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)
    document_id   = Column(Integer, ForeignKey('document.id'))
    document      = relationship('Document', backref=backref('phrases', cascade='all, delete-orphan'), foreign_keys=document_id)
    phrase_num    = Column(Integer, nullable=False) # unique Phrase number per document
    text          = Column(Text, nullable=False)
    words         = Column(STR_ARRAY_TYPE)
    char_offsets  = Column(INT_ARRAY_TYPE)
    entity_cids   = Column(STR_ARRAY_TYPE)
    entity_types  = Column(STR_ARRAY_TYPE)

    __mapper_args__ = {
        'polymorphic_identity': 'phrase',
    }

    __table_args__ = (
        UniqueConstraint(document_id, phrase_num),
    )

    def __repr__(self):
        if self.is_tabular():
            rows = tuple([self.row_start, self.row_end]) \
                   if self.row_start != self.row_end else self.row_start
            cols = tuple([self.col_start, self.col_end]) \
                   if self.col_start != self.col_end else self.col_start
            return ("Phrase (Doc: %s, Table: %s, Row: %s, Col: %s, Index: %s, Text: %s)" %
                   (self.document.name.encode('utf-8'),
                    self.table.position,
                    rows,
                    cols,
                    self.position,
                    self.text.encode('utf-8')))
        else:
            return ("Phrase (Doc: %s, Index: %s, Text: %s)" %
                (self.document.name.encode('utf-8'),
                self.phrase_num,
                self.text.encode('utf-8')))

    def _asdict(self):
        return {
            # base
            'id': self.id,
            # 'document': self.document,
            'phrase_num': self.phrase_num,
            'text': self.text,
            'entity_cids': self.entity_cids,
            'entity_types': self.entity_types,
            # tabular
            # 'table': self.table,
            # 'cell': self.cell,
            'row_start': self.row_start,
            'row_end': self.row_end,
            'col_start': self.col_start,
            'col_end': self.col_end,
            'position': self.position,
            # lingual
            'words': self.words,
            'char_offsets': self.char_offsets,
            'lemmas': self.lemmas,
            'pos_tags': self.pos_tags,
            'ner_tags': self.ner_tags,
            'dep_parents': self.dep_parents,
            'dep_labels': self.dep_labels,
            # visual
            'page': self.page,
            'top': self.top,
            'bottom': self.bottom,
            'left': self.left,
            'right': self.right
        }

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()


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
                for (key, val) in insert_args.items():
                    if isinstance(val, list):
                        if snorkel_postgres:
                            insert_args[key] = val
                        else:
                            insert_args[key] = pickle.dumps(val) #NOTE: this works for sqlite, not Postgres
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
        self.sentence   = sentence  # The sentence Context of the Span
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

    def is_lingual(self):
        return self.sentence.is_lingual()

    def is_structural(self):
        return self.sentence.is_structural()

    def is_visual(self):
        return self.sentence.is_visual()

    def is_tabular(self):
        return self.sentence.is_tabular()

    def __contains__(self, other_span):
        return (self.sentence == other_span.sentence
            and other_span.char_start >= self.char_start
            and other_span.char_end <= self.char_end)

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
    id            = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)
    sentence_id   = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'))
    char_start    = Column(Integer, nullable=False)
    char_end      = Column(Integer, nullable=False)
    meta          = Column(PickleType)

    __table_args__ = (
        UniqueConstraint(sentence_id, char_start, char_end),
    )

    __mapper_args__ = {
        'polymorphic_identity': 'span',
        'inherit_condition': (id == Context.id)
    }

    sentence = relationship('Context', backref=backref('spans', cascade='all, delete-orphan'), foreign_keys=sentence_id)

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


class TemporaryImplicitSpan(TemporarySpan):
    """The TemporaryContext version of ImplicitSpan"""
    def __init__(self, sentence, char_start, char_end, expander_key, position,
            text, words, lemmas, pos_tags, ner_tags, dep_parents, dep_labels,
            page, top, left, bottom, right, meta=None):
        super(TemporarySpan, self).__init__()
        self.sentence       = sentence  # The sentence Context of the Span
        self.char_start     = char_start
        self.char_end       = char_end
        self.expander_key   = expander_key
        self.position       = position
        self.text           = text
        self.words          = words
        self.lemmas         = lemmas
        self.pos_tags       = pos_tags
        self.ner_tags       = ner_tags
        self.dep_parents    = dep_parents
        self.dep_labels     = dep_labels
        self.page           = page
        self.top            = top
        self.left           = left
        self.bottom         = bottom
        self.right          = right
        self.meta           = meta

    def __len__(self):
        return sum(map(len, words))

    def __eq__(self, other):
        try:
            return (self.sentence == other.sentence and
                    self.char_start == other.char_start and
                    self.char_end == other.char_end and
                    self.expander_key == other.expander_key and
                    self.position == other.position)
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return (self.sentence != other.sentence or
                    self.char_start != other.char_start or
                    self.char_end != other.char_end or
                    self.expander_key != other.expander_key or
                    self.position != other.position)
        except AttributeError:
            return True

    def __hash__(self):
        return (hash(self.sentence) + hash(self.char_start) + hash(self.char_end)
                + hash(self.expander_key) + hash(self.position))

    def get_stable_id(self):
        # return (construct_stable_id(self.sentence, self._get_polymorphic_identity(), self.char_start, self.char_end)
        #     + ':%s:%s' % (self.expander_key, self.position))
        return '%s::%s:%s:%s:%s:%s:%s' % (
            self.sentence.document.name,
            self._get_polymorphic_identity(),
            self.sentence.id,
            self.char_start,
            self.char_end,
            self.expander_key,
            self.position)

    def _get_table_name(self):
        return 'implicit_span'

    def _get_polymorphic_identity(self):
        return 'implicit_span'

    def _get_insert_query(self):
        return """INSERT INTO implicit_span VALUES(
            :id,
            :sentence_id,
            :char_start,
            :char_end,
            :expander_key,
            :position,
            :text,
            :words,
            :lemmas,
            :pos_tags,
            :ner_tags,
            :dep_parents,
            :dep_labels,
            :page,
            :top,
            :left,
            :bottom,
            :right,
            :meta)"""

    def _get_insert_args(self):
        return {'sentence_id'   : self.sentence.id,
                'char_start'    : self.char_start,
                'char_end'      : self.char_end,
                'expander_key'  : self.expander_key,
                'position'      : self.position,
                'text'          : self.text,
                'words'         : self.words,
                'lemmas'        : self.lemmas,
                'pos_tags'      : self.pos_tags,
                'ner_tags'      : self.ner_tags,
                'dep_parents'   : self.dep_parents,
                'dep_labels'    : self.dep_labels,
                'page'          : self.page,
                'top'           : self.top,
                'left'          : self.left,
                'bottom'        : self.bottom,
                'right'         : self.right,
                'meta'          : self.meta
                }

    def get_attrib_tokens(self, a='words'):
        """Get the tokens of sentence attribute _a_ over the range defined by word_offset, n"""
        return self.__getattribute__(a)

    def get_attrib_span(self, a, sep=" "):
        """Get the span of sentence attribute _a_ over the range defined by word_offset, n"""
        if a == 'words':
            return self.text
        else:
            return sep.join(self.get_attrib_tokens(a))

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
            return self._get_instance(sentence=self.sentence, char_start=char_start, char_end=char_end, expander_key=expander_key,
                position=position, text=text, words=words, lemmas=lemmas, pos_tags=pos_tags,
                ner_tags=ner_tags, dep_parents=dep_parents, dep_labels=dep_labels,
                page=page, top=top, left=left, bottom=bottom, right=right, meta=meta)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return '%s("%s", sentence=%s, words=[%s,%s], position=[%s])' \
            % (self.__class__.__name__, self.get_span().encode('utf-8'), self.sentence.id,
               self.get_word_start(), self.get_word_end(), self.position)

    def _get_instance(self, **kwargs):
        return TemporaryImplicitSpan(**kwargs)


class ImplicitSpan(Context, TemporaryImplicitSpan):
    """
    A span of characters that may not have appeared verbatim in the source text.
    It is identified by Context id, character-index start and end (inclusive),
    as well as a key representing what 'expander' function drew the ImplicitSpan
    from an  existing Span, and a position (where position=0 corresponds to the
    first ImplicitSpan produced from the expander function).

    The character-index start and end point to the segment of text that was
    expanded to produce the ImplicitSpan.
    """
    __tablename__ = 'implicit_span'
    id            = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)
    sentence_id   = Column(Integer, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)
    char_start    = Column(Integer, nullable=False)
    char_end      = Column(Integer, nullable=False)
    expander_key  = Column(String, nullable=False)
    position      = Column(Integer, nullable=False)
    text          = Column(String)
    if snorkel_postgres:
        words       = Column(postgresql.ARRAY(String), nullable=False)
        lemmas      = Column(postgresql.ARRAY(String))
        pos_tags    = Column(postgresql.ARRAY(String))
        ner_tags    = Column(postgresql.ARRAY(String))
        dep_parents = Column(postgresql.ARRAY(Integer))
        dep_labels  = Column(postgresql.ARRAY(String))
        page        = Column(postgresql.ARRAY(Integer))
        top         = Column(postgresql.ARRAY(Integer))
        left        = Column(postgresql.ARRAY(Integer))
        bottom      = Column(postgresql.ARRAY(Integer))
        right       = Column(postgresql.ARRAY(Integer))
    else:
        words       = Column(PickleType, nullable=False)
        lemmas      = Column(PickleType)
        pos_tags    = Column(PickleType)
        ner_tags    = Column(PickleType)
        dep_parents = Column(PickleType)
        dep_labels  = Column(PickleType)
        page        = Column(PickleType)
        top         = Column(PickleType)
        left        = Column(PickleType)
        bottom      = Column(PickleType)
        right       = Column(PickleType)
    meta = Column(PickleType)

    __table_args__ = (
        UniqueConstraint(sentence_id, char_start, char_end, expander_key, position),
    )

    __mapper_args__ = {
        'polymorphic_identity': 'implicit_span',
        'inherit_condition': (id == Context.id)
    }

    sentence = relationship('Context', backref=backref('implicit_spans', cascade='all, delete-orphan'), foreign_keys=sentence_id)

    def _get_instance(self, **kwargs):
        return ImplicitSpan(**kwargs)

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
