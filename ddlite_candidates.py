

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
    def __init__(self, n, sentence_id, char_offset, char_len, words, sep=" ", word_offset=None):
        self.id          = "%s_%s_%s" % (sentence_id, char_offset, char_len)
        self.n           = n
        self.sentence_id = sentence_id
        self.char_offset = char_offset
        self.char_len    = char_len
        self.word_offset = word_offset
        self.words       = words
        
        # Get the actual span indicated by char_offset and char_len
        s           = sep.join(words) if isinstance(sep, str) else "".join(zip(words, sep))
        self.string = s[:char_len]


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
            s_id    = d[id]
            s_words = d[words_key]
            s_cos   = d[char_offsets_key]
        except:
            print "Input object must have id, words and char_offset attributes:"
            print "\n\twords_key='%s'\n\tchar_offsets_key='%s'" % (words_key, char_offsets_key)
            raise ValueError()

        # Loop over all n-grams in **reverse** order (to facilitate longest-match semantics)
        L = len(words)
        for l in range(1, self.n_max+1)[::-1]:
            for i in range(L-l+1):

                # Use the full token spans--**with default separator**--by default
                co    = s_cos[i]
                words = [s_words[i] for i in range(i, i+l)]
                cl    = len(' '.join(words))
                yield Ngram(n=l, sentence_id=s_id, char_offset=co, char_len=cl, words=words, \
                        word_offset=i)
