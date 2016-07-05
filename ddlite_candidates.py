

class Candidate(object):
    """A candidate object, **uniquely identified by its id**"""
    def __init__(self, id):
        self.id = id
    
    def __eq__(self, other):
        try:
            return self.id == other.id
        except:
            raise NotImplementedError()
    
    def __hash__(self):
        return hash(self.id)


class CandidateSpace(object):
    """
    Defines the **space** of candidate objects
    Calling _apply(x)_ given an object _x_ returns a generator over candidates in _x_.
    """
    def __init__(self):
        pass

    def apply(self, x):
        raise NotImplementedError()


# Basic sentence attributes
WORDS        = 'words'
CHAR_OFFSETS = 'char_offsets'
TEXT         = 'text'

class Ngram(Candidate):
    """A span of _n_ tokens, identified by sentence id and character-index start, end (inclusive)"""
    def __init__(self, char_start, char_end, sent, metadata={}):
        
        # Inherit full sentence object (tranformed to dict) and check for necessary attribs
        self.sentence = sent if isinstance(sent, dict) else sent._asdict()
        self.sent_id  = self.sentence['id']
        REQ_ATTRIBS = ['id', WORDS]
        if not all([self.sentence.has_key(a) for a in REQ_ATTRIBS]):
            raise ValueError("Sentence object must have attributes %s to form Ngram object" % ", ".join(REQ_ATTRIBS))

        # Set basic object attributes
        self.id          = "%s:%s-%s" % (self.sent_id, char_start, char_end)
        self.char_start  = char_start
        self.char_end    = char_end
        self.char_len    = char_end - char_start + 1
        self.word_start  = self.char_to_word_index(char_start)
        self.word_end    = self.char_to_word_index(char_end)
        self.n           = self.word_end - self.word_start + 1

        # NOTE: We assume that the char_offsets are **relative to the document start**
        self.sent_offset     = self.sentence[CHAR_OFFSETS][0]
        self.sent_char_start = self.char_start - self.sent_offset
        self.sent_char_end   = self.char_end - self.sent_offset

        # A dictionary to hold task-specific metadata e.g. canonical id, category, etc.
        self.metadata = metadata

    def char_to_word_index(self, ci):
        """Given a character-level index (offset), return the index of the **word this char is in**"""
        for i,co in enumerate(self.sentence[CHAR_OFFSETS]):
            if ci == co:
                return i
            elif ci < co:
                return i-1
        return i

    def word_to_char_index(self, wi):
        """Given a word-level index, return the character-level index (offset) of the word's start"""
        return self.sentence[CHAR_OFFSETS][wi]
    
    def get_attrib_tokens(self, a):
        """Get the tokens of sentence attribute _a_ over the range defined by word_offset, n"""
        return self.sentence[a][self.word_start:self.word_end+1]
    
    def get_attrib_span(self, a, sep=" "):
        """Get the span of sentence attribute _a_ over the range defined by word_offset, n"""
        # NOTE: Special behavior for words currently (due to correspondence with char_offsets)
        if a == WORDS:
            return self.sentence[TEXT][self.sent_char_start:self.sent_char_end+1]
        else:
            return sep.join(self.get_attrib_tokens(a))
    
    def get_span(self, sep=" "):
        return self.get_attrib_span(WORDS)

    def __getitem__(self, key):
        """
        Slice operation returns a new candidate sliced according to **char index**
        Note that the slicing is w.r.t. the candidate range (not the abs. sentence char indexing)
        """
        if isinstance(key, slice):
            char_start = self.char_start if key.start is None else self.char_start + key.start
            char_end   = self.char_end if key.stop is None else self.char_start + key.stop - 1
            return Ngram(char_start, char_end, self.sentence)
        else:
            raise NotImplementedError()
        
    def __repr__(self):
        return '<Ngram("%s", id=%s, chars=[%s,%s], words=[%s,%s])' \
            % (self.get_attrib_span(WORDS), self.id, self.char_start, self.char_end, self.word_start, self.word_end)


class Ngrams(CandidateSpace):
    """
    Defines the space of candidates as all n-grams (n <= n_max) in a sentence _x_,
    indexing by **character offset**.
    """
    def __init__(self, n_max=5):
        self.n_max = n_max
    
    def apply(self, x):
        s = x if isinstance(x, dict) else x._asdict()
        try:
            cos   = s[CHAR_OFFSETS]
            words = s[WORDS]
        except:
            raise ValueError("Input object must have %s, %s attributes" % (CHAR_OFFSET, WORDS))

        # Loop over all n-grams in **reverse** order (to facilitate longest-match semantics)
        L = len(cos)
        for l in range(1, self.n_max+1)[::-1]:
            for i in range(L-l+1):
                ws = words[i:i+l] 
                cl = cos[i+l-1] - cos[i] + len(words[i+l-1])  # NOTE that we derive char_len without using sep
                yield Ngram(char_start=cos[i], char_end=cos[i]+cl-1, sent=s)
