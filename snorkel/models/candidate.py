from .meta import SnorkelBase, snorkel_postgres
from sqlalchemy import Table, Column, String, Integer, Text, ForeignKey, ForeignKeyConstraint, Index
from sqlalchemy.orm import relationship, backref
from sqlalchemy.types import PickleType
from snorkel.utils import slice_into_ngrams

class CandidateSet(SnorkelBase):
    """A named collection of Candidate objects."""
    __tablename__ = 'candidate_set'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    def __repr__(self):
        return "Candidate Set (" + str(self.name) + ")"

    def append(self, item):
        self.candidates.append(item)

    def remove(self, item):
        self.candidates.remove(item)

    def __iter__(self):
        """Default iterator is over Candidate objects"""
        for candidate in self.candidates:
            yield candidate

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, key):
        return self.candidates[key]


class Candidate(SnorkelBase):
    """
    A candidate relation.
    """
    __tablename__ = 'candidate'
    id = Column(Integer, primary_key=True)
    candidate_set_id = Column(Integer, ForeignKey('candidate_set.id'))
    set = relationship('CandidateSet', backref=backref('candidates', cascade='all, delete-orphan'))
    type = Column(String, nullable=False)
    # TEMP #
    uid = Column(String, nullable=False)

    # Postgres requires an explicit unique index to use a tuple as a compound foreign key,
    # even if it is redundant as in this case
    if snorkel_postgres:
        __table_args__ = (
            Index('candidate_unique_id_candidate_set_id', id, candidate_set_id, unique=True),
        )

    __mapper_args__ = {
        'polymorphic_identity': 'candidate',
        'polymorphic_on': type
    }


class TemporarySpan(object):

    def __init__(self, context=None, char_end=None, char_start=None, uid=None):
        self.context = context
        self.char_end = char_end
        self.char_start = char_start
        # TEMP
        self.uid = uid
        # ALSO TEMP is the uid keyword argument to __init__

    def __len__(self):
        return self.char_end - self.char_start + 1

    def __eq__(self, other):
        if isinstance(other, TemporarySpan):
            return self.context == other.context and self.char_start == other.char_start and self.char_end == other.char_end
        else:
            return False

    def __hash__(self):
        return hash(self.context) + hash(self.char_start) + hash(self.char_end)

    def get_word_start(self):
        return self.char_to_word_index(self.char_start)

    def get_word_end(self):
        return self.char_to_word_index(self.char_end)

    def get_n(self):
        return self.get_word_end() - self.get_word_start() + 1

    def char_to_word_index(self, ci):
        """Given a character-level index (offset), return the index of the **word this char is in**"""
        i = None
        for i, co in enumerate(self.context.char_offsets):
            if ci == co:
                return i
            elif ci < co:
                return i-1
        return i

    def word_to_char_index(self, wi):
        """Given a word-level index, return the character-level index (offset) of the word's start"""
        return self.context.char_offsets[wi]

    def get_attrib_tokens(self, a):
        """Get the tokens of sentence attribute _a_ over the range defined by word_offset, n"""
        return self.context.__getattribute__(a)[self.get_word_start():self.get_word_end() + 1]

    def get_attrib_span(self, a, sep=" "):
        """Get the span of sentence attribute _a_ over the range defined by word_offset, n"""
        # NOTE: Special behavior for words currently (due to correspondence with char_offsets)
        if a == 'words':
            return self.context.text[self.char_start:self.char_end + 1]
        else:
            return sep.join(self.get_attrib_tokens(a))

    def get_span(self, sep=" "):
        return self.get_attrib_span('words', sep)

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
            return self._get_instance(char_start=char_start, char_end=char_end, context=self.context)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return '%s("%s", context=%s, chars=[%s,%s], words=[%s,%s])' \
            % (self.__class__.__name__, self.get_span(), self.context.id, self.char_start, self.char_end,
               self.get_word_start(), self.get_word_end())

    def _get_instance(self, **kwargs):
        return TemporarySpan(**kwargs)

    def promote(self):
        # TEMP uid
        return Span(context=self.context, char_start=self.char_start, char_end=self.char_end, uid=self.uid)


class Span(Candidate, TemporarySpan):
    """
    A span of characters, identified by Context id and character-index start, end (inclusive).

    char_offsets are **relative to the Context start**
    """
    __table__ = Table('span', SnorkelBase.metadata,
                      Column('id', Integer, unique=True),
                      Column('candidate_set_id', Integer, primary_key=True),
                      Column('context_id', Integer, ForeignKey('context.id'), primary_key=True),
                      Column('char_start', Integer, primary_key=True),
                      Column('char_end', Integer, primary_key=True),
                      Column('meta', PickleType),
                      ForeignKeyConstraint(['id', 'candidate_set_id'], ['candidate.id', 'candidate.candidate_set_id'])
                      )

    # Postgres requires an explicit unique index to use a tuple as a compound foreign key,
    # even if it is redundant as in this case
    if snorkel_postgres:
        __table_args__ = (
            Index('span_unique_id_candidate_set_id', __table__.c.id, __table__.c.candidate_set_id, unique=True),
        )

    context = relationship('Context', backref=backref('candidates', cascade_backrefs=False))

    __mapper_args__ = {
        'polymorphic_identity': 'span',
    }

    def _get_instance(self, **kwargs):
        return Span(**kwargs)

    ### LF Utilities ###
    def post_window(self, attr='words'):
        return getattr(self.context, attr)[self.get_word_start()+1:]

    def pre_window(self, attr='words'):
        return getattr(self.context, attr)[:self.get_word_start()]

    def post_ngrams(self, attr='words'):
        return self.post_window(attr=attr)

    def pre_ngrams(self, attr='words'):
        return self.pre_window(attr=attr)

    def cell_ngrams(self, attr='words'):
        return self.post_ngrams(attr=attr) + self.pre_ngrams(attr=attr)

    def neighborhood_ngrams(self, attr='words', n_max=3, dist=1, case_sensitive=False):
        # TODO: optimize with SQL query
        f = lambda x: 0 < x and x <= dist
        phrases = [phrase for phrase in self.context.table.phrases if
            f(abs(phrase.row_num - self.context.row_num) + abs(phrase.col_num - self.context.col_num))]
        for phrase in phrases:
            for ngram in slice_into_ngrams(getattr(phrase,attr), n_max=n_max):
                yield ngram if case_sensitive else ngram.lower()

    def neighbor_ngrams(self, attr='words', n_max=3, dist=1, case_sensitive=False):
        # TODO: optimize with SQL query
        for phrase in self.context.table.phrases:
            side = ''
            row_diff = phrase.row_num - self.context.row_num
            col_diff = phrase.col_num - self.context.col_num
            if col_diff==0:
                if 0 < row_diff and row_diff <= dist:
                    side = "UP"
                elif  0 > row_diff and row_diff >= -dist:
                    side = "DOWN"
            elif row_diff==0:
                if 0 < col_diff and col_diff <= dist:
                    side = "RIGHT"
                elif  0 > col_diff and col_diff >= -dist:
                    side = "LEFT"
            if side:
                for ngram in slice_into_ngrams(getattr(phrase,attr), n_max=n_max):
                    yield (ngram,side) if case_sensitive else (ngram.lower(), side)


    def aligned_ngrams(self, attr='words', n_max=3, case_sensitive=False):
        return (self.row_ngrams(attr=attr, n_max=n_max, case_sensitive=case_sensitive)
              + self.col_ngrams(attr=attr, n_max=n_max, case_sensitive=case_sensitive))

    def row_ngrams(self, attr='words', n_max=3, case_sensitive=False):
        ngrams = [ngram for ngram in self._get_aligned_ngrams(attr=attr, n_max=n_max, axis='row')]
        return [ngram.lower() for ngram in ngrams] if not case_sensitive else ngrams

    def col_ngrams(self, attr='words', n_max=3, case_sensitive=False):
        ngrams = [ngram for ngram in self._get_aligned_ngrams(attr=attr, n_max=n_max, axis='col')]
        return [ngram.lower() for ngram in ngrams] if not case_sensitive else ngrams

    def _get_aligned_ngrams(self, n_max=3, attr='words', axis='row'):
        axis_name = axis + '_num'
        phrases = [phrase for phrase in self.context.table.phrases
            if getattr(phrase,axis_name) == getattr(self.context,axis_name)
            and phrase != self.context]
        for phrase in phrases:
            for ngram in slice_into_ngrams(getattr(phrase,attr), n_max=n_max):
                yield ngram


class SpanPair(Candidate):
    """
    A pair of Span Candidates, representing a relation from Span 0 to Span 1.
    """
    __table__ = Table('span_pair', SnorkelBase.metadata,
                      Column('id', Integer, unique=True),
                      Column('candidate_set_id', Integer, primary_key=True),
                      Column('span0_id', Integer, primary_key=True),
                      Column('span1_id', Integer, primary_key=True),
                      ForeignKeyConstraint(['id', 'candidate_set_id'], ['candidate.id', 'candidate.candidate_set_id']),
                      ForeignKeyConstraint(['span0_id', 'candidate_set_id'], ['span.id', 'span.candidate_set_id']),
                      ForeignKeyConstraint(['span1_id', 'candidate_set_id'], ['span.id', 'span.candidate_set_id'])
                      )

    span0 = relationship('Span', backref=backref('span_source_pairs', cascade_backrefs=False),
                          cascade_backrefs=False, foreign_keys='SpanPair.span0_id')
    span1 = relationship('Span', backref=backref('span_dest_pairs', cascade_backrefs=False),
                          cascade_backrefs=False, foreign_keys='SpanPair.span1_id')

    __mapper_args__ = {
        'polymorphic_identity': 'span_pair',
    }

    def __getitem__(self, key):
        if key == 0:
            return self.span0
        elif key == 1:
            return self.span1
        else:
            raise KeyError('Valid keys are 0 and 1.')

    def __repr__(self):
        return "SpanPair(%s, %s)" % (self.span0, self.span1)

    #TEMP
    def promote(self):
        return self
