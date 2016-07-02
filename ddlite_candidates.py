

class Candidate(object):
    """A candidate object, to be classified."""
    pass


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


class Ngram(Candidate):
    """A span of _n_ tokens, identified by sentence id and character-index start, end (inclusive)"""
    def __init__(self, char_start, char_end, word_start, word_end, sent):
        
        # Inherit full sentence object (tranformed to dict) and check for necessary attribs
        self.s = sent if isinstance(sent, dict) else sent._asdict()
        REQ_ATTRIBS = ['id', WORDS]
        if not all([self.s.has_key(a) for a in REQ_ATTRIBS]):
            raise ValueError("Sentence object must have attributes %s to form Ngram object" % ", ".join(REQ_ATTRIBS))

        # Set basic object attributes
        self.id          = "%s_chars:%s-%s" % (self.s['id'], char_start, char_end)
        self.char_start  = char_start
        self.char_end    = char_end
        self.char_len    = char_end - char_start + 1
        self.word_start  = word_start
        self.word_end    = word_end
        self.n           = word_end - word_start + 1
    
    def get_attrib_tokens(self, a):
        """Get the tokens of sentence attribute _a_ over the range defined by word_offset, n"""
        return self.s[a][self.word_start:self.word_end+1]
    
    def get_attrib_span(self, a, sep=" "):
        """Get the span of sentence attribute _a_ over the range defined by word_offset, n"""
        span = sep.join(self.get_attrib_tokens(a))

        # NOTE: Special behavior for words currently (due to correspondence with char_offsets)
        if a == WORDS:
            return span[:self.char_len]
        else:
            return span
        
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
                cl = len(' '.join(ws))  # NOTE: Use full ' '-separated word spans by default
                yield Ngram(char_start=cos[i], char_end=cos[i]+cl-1, word_start=i, word_end=i+l-1, sent=s)
