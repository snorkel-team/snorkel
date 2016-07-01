

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


class Ngram(Candidate):
    """A span of _n_ tokens, identified by sentence id and character offset."""
    def __init__(self, n, sentence_id, char_offset, words, sep=" ", word_offset=None):
        self.id          = "%s_%s_%s" % (sentence_id, char_offset, n)
        self.n           = n
        self.sentence_id = sentence_id
        self.char_offset = char_offset
        self.word_offset = word_offset
        self.words       = words
        self.string      = sep.join(words) if isinstance(sep, str) else "".join(zip(words, sep))


class Ngrams(CandidateSpace):
    """
    Defines the space of candidates as all n-grams (n <= n_max) in a sentence _x_,
    indexing by **character offset**.
    """
    def __init__(self, n_max=5, words_key='words', char_offsets_key='char_offsets'):
        self.n_max         = n_max
        self.words_key     = 'words'
        self.char_idxs_key = 'char_offsets'
    
    def apply(self, x):
        d = x if isinstance(x, dict) else x._asdict()
        try:
            sid          = d[id]
            words        = d[words_key]
            char_offsets = d[char_offsets_key]
        except:
            print "Input object must have id, words and char_offset attributes:"
            print "\n\twords_key='%s'\n\tchar_offsets_key='%s'" % (words_key, char_offsets_key)
            raise ValueError()

        # Loop over all n-grams in **reverse** order (to facilitate longest-match semantics)
        L = len(words)
        for l in range(1, self.n_max+1)[::-1]:
            for i in range(L-l+1):
                yield Ngram(n=l, sentence_id=sid, char_offset=char_offsets[i], \
                        words=[words[i] for i in range(i, i+l)], word_offset=i)
