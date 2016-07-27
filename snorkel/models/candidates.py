from .meta import SnorkelBase
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.types import PickleType


class Candidates(SnorkelBase):
    """A named collection of Candidate objects."""
    __tablename__ = 'candidates'
    id = Column(String, primary_key=True)

    candidates = relationship('Candidate', backref='candidates')

    def __repr__(self):
        return "Candidates" + str((self.id,))

    def __iter__(self):
        """Default iterator is over Candidate objects"""
        for candidate in self.candidates:
            yield candidate

    def __len__(self):
        return len(self.candidates)


class Candidate(SnorkelBase):
    """
    A candidate k-arity relation, **uniquely identified by its id**.
    """
    __tablename__ = 'candidate'
    id = Column(String, primary_key=True)
    type = Column(String)
    candidates_id = Column(String, ForeignKey('candidates.id'))
    context_id = Column(String, ForeignKey('context.id'))

    __mapper_args__ = {
        'polymorphic_identity': 'candidate',
        'polymorphic_on': type
    }


class Ngram(Candidate):
    """
    A span of _n_ tokens, identified by Context id and character-index start, end (inclusive).

    char_offsets are **relative to the Document start**
    """
    __tablename__ = 'ngram'
    id = Column(String, ForeignKey('candidate.id'), primary_key=True)
    char_start = Column(Integer)
    char_end = Column(Integer)
    meta = Column(PickleType)

    __mapper_args__ = {
        'polymorphic_identity': 'ngram',
    }

    def __init__(self, char_start, char_end, context, meta=None):
        self.id = "ngram-%s-%s-%s" % (context.id, str(char_start), str(char_end))
        self.char_start = char_start
        self.char_end = char_end
        self.context = context
        if meta is not None:
            self.meta = meta

    def __len__(self):
        return self.char_end - self.char_start + 1

    # TODO: Below methods could be replaced with transient members, i.e., not persisted, using the @reconstructor decorator

    def get_word_start(self):
        return self.char_to_word_index(self.char_start)

    def get_word_end(self):
        return self.char_to_word_index(self.char_end)

    def get_n(self):
        return self.get_word_end() - self.get_word_start() + 1

    def get_sent_offset(self):
        return self.context.char_offsets[0]

    def get_sent_char_start(self):
        return self.char_start - self.get_sent_offset()

    def get_sent_char_end(self):
        return self.char_end - self.get_sent_offset()

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
            return self.context.text[self.get_sent_char_start():self.get_sent_char_end() + 1]
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
            return Ngram(char_start, char_end, self.context)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return 'Ngram("%s", id=%s, chars=[%s,%s], words=[%s,%s])' \
            % (" ".join(self.context.words), self.id, self.char_start, self.char_end, self.get_word_start(), self.get_word_end())
