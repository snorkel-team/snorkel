from collections import defaultdict
import sys
import re
import numpy as np
from scipy.sparse import lil_matrix, identity
from sklearn.preprocessing import normalize
import random
from nltk.stem.porter import PorterStemmer


DEFAULT_IDF = 1.0

class Vectorizer(object):
    """
    Generic class for vectorizing;by default, initializes over a list of phrases, and does not TF-IDF weighting
    """
    def __init__(self, phrases, other_phrases=None, stem=True, stopwords=set(), rgx_filter=r'\w+', min_len=1):
        self.stem = stem
        if self.stem:
            self.stemmer = PorterStemmer()
        self.stopwords  = stopwords
        self.rgx_filter = rgx_filter
        self.min_len    = min_len

        self.word_index     = {}
        self.inv_word_index = {}
        self.word_idf       = defaultdict(lambda : DEFAULT_IDF)
        self.word_to_cids   = defaultdict(set)

        self._compile_dicts(phrases, other_phrases)

    def _compile_dicts(self, phrases, other_phrases):

        # Process the canonical dictionary cd
        for phrase in phrases:
            for word in self._split(phrase):

                # Add to word index
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)

        # Construct inverted word index
        for word, i in self.word_index.iteritems():
            self.inv_word_index[i] = word

        print "|V| = %s" % len(self.word_index)

    def vectorize_phrases(self, phrases):
        """
        Transform a list of N phrases into a N x |V|+1 row-normalized sparse matrix
        where each row represents a TF-IDF vectorized phrase.
        NOTE: Reserve the last column for "UNK", i.e. OOV words
        """
        V = len(self.word_index)
        X = lil_matrix((len(phrases), V + 1))
        for i, phrase in enumerate(phrases):
            for word in self._split(phrase):
                j = self.word_index[word] if word in self.word_index else V
                s = self.word_idf[word] if word in self.word_idf else DEFAULT_IDF
                X[i, j] += s
        X = X.tocsr()
        return normalize(X)
    
    def _split(self, phrase):
        for w in re.findall(self.rgx_filter, phrase.lower()):
            if w not in self.stopwords and len(w) > self.min_len:
                yield self.stemmer.stem(w) if self.stem else w


class TFIDFVectorizer(Vectorizer):
    """Basic TF_IDF extension"""
    def _compile_dicts(self, phrases, other_phrases):
        word_to_phrases = defaultdict(set)
        for i, phrase in enumerate(phrases):
            for word in self._split(phrase):

                # Add to word index
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)

                word_to_phrases[word].add(i)

        # Compute word IDF scores **based on canonical ID not term counts!**
        N = len(phrases)
        for word in self.word_index.keys():
            self.word_idf[word] = np.log( N / float(len(word_to_phrases[word])) )

        for phrase in other_phrases:
            for word in self._split(phrase):
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)

                    # TODO: Should IDF = 0.0 by default?
                    self.word_idf[word] = 0.0

        # Construct inverted word index
        for word, i in self.word_index.iteritems():
            self.inv_word_index[i] = word

        print "|V| = %s" % len(self.word_index)

class CanonDictVectorizer(Vectorizer):
    """
    Class for vectorizing phrases using TF-IDF scheme based on a _canonical dictionary_.
    The canonical dictionary has a set of _terms_ which each map to a _canonical ID_;
    IDF is computed **at the canonical ID level**.
    """
    def _compile_dicts(self, cd, other_phrases):
        
        # Process the canonical dictionary cd
        all_cids = set()
        for term, cids in cd.iteritems():
            for cid in cids:
                all_cids.add(cid)

            for word in self._split(term):

                # Add to word index
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)

                # Add cid to word_to_cid dict
                for cid in cids:
                    self.word_to_cids[word].add(cid)

        # Compute word IDF scores **based on canonical ID not term counts!**
        N = len(all_cids)
        for word in self.word_index.keys():
            if len(self.word_to_cids[word]) > 0:
                self.word_idf[word] = np.log( N / float(len(self.word_to_cids[word])) )

        # Add other OOD phrases, e.g. from training set
        # TODO: How should these be weighted?
        for phrase in other_phrases:
            for word in self._split(phrase):
                if word not in self.word_index:
                    self.word_idf[word]   = DEFAULT_IDF
                    self.word_index[word] = len(self.word_index)

        # Construct inverted word index
        for word, i in self.word_index.iteritems():
            self.inv_word_index[i] = word

        print "|V| = %s" % len(self.word_index)


class SSIModel(object):
    def __init__(self, D, cid_sets):
        """
        D is an M x V dictionary term matrix, where V is the vocabulary size
        cid_sets is a corresponding M-length list of *sets* of CIDs
        """
        self.D        = D
        self.cid_sets = cid_sets
        
        #self.cids      = frozenset(d)
        self.T, self.V = D.shape
        
        # Index the dictionary terms (D) by cid (d)
        self.cid_to_rows = defaultdict(set)
        for i, cids in enumerate(self.cid_sets):
            for cid in cids:
                self.cid_to_rows[cid].add(i)
        
        # Store params from each iter
        self.Ws = []
        self.bs = []
    
    def train(self, X, Y, rate=1e-3, n_iter=3, n_iter_sample=10, verbose=True):
        """
        X is an N x V sparse matrix of N vectorized disease mentions (V = vocab size) to be linked to CIDs
        Y is an N x K sparse matrix of label *probabilities*, where K = |CIDs|
        """
        self.Ws = []
        N       = X.shape[0]

        # Initialize W = I
        W = identity(self.V, format='csr')

        # Cosine similarity term
        b = 1.0

        # Run SGD
        for it in range(n_iter):
            print "Iteration: %s" % it

            # Iterate in random order through the training examples
            run = range(N)
            random.shuffle(run)
            for count, i in enumerate(run):
                if count % 250 == 0:
                    sys.stdout.write("\r\t%s" % count)

                # Randomly pick a tuple (x, t^+, t^-)
                # First pick x, a random training phrase
                x = X.getrow(i)

                # Precompute the matches in the outer loop here
                #matches = Z.data.argsort()
                Z = self.D * x.T

                # Sample the training CID label according to the training distribution from the gen. model
                for si in range(n_iter_sample):
                    cids = Y.getrow(i).nonzero()[1]
                    t    = random.random()
                    for cid in cids:
                        t -= Y[i,cid]
                        if t < 0:
                            break

                    # Skip OOD training examples
                    if cid not in self.cid_to_rows:
                        continue

                    # Of the terms in the dictionary corresponding to the label CID, pick the one
                    # closest to x
                    ps = list(self.cid_to_rows[cid])
                    Zp = Z[ps]
                    if Zp.nnz == 0:
                        p = random.choice(ps)
                    else:
                        Zp = Zp.tocoo()
                        p  = ps[Zp.row[Zp.data.argmax()]]
                    tp = self.D.getrow(p)

                    # Of the other terms in the dictionary, pick the closest match as negative example
                    ns = list(set(range(self.T)).difference(ps))
                    Zn = Z[ns]
                    if Zn.nnz == 0:
                        n = random.choice(ns)
                    else:
                        Zn = Zn.tocoo()
                        n  = ns[Zn.row[Zn.data.argmax()]]
                    tn = self.D.getrow(n)

                    # Take gradient step
                    xtp = (x * tp.T)[0,0]
                    xtn = (x * tn.T)[0,0]
                    xw  = x * W
                    d = 1 - (xw * tp.T)[0,0] - b * xtp + (xw * tn.T)[0,0] + b * xtn
                    """
                    print "\n"
                    print "d=", d
                    print "i:", i
                    print "cid:", cid
                    print "p:", p
                    print "n:", n
                    """
                    if d > 0:
                        W = W + rate * (x.T * tp - x.T * tn)
                        b = b + rate * (xtp - xtn)

            self.Ws.append(W)
            self.bs.append(b)

        print "\n"
        self.W = W
        self.b = b

    def predict(self, x, W=None, b=None):
        """
        Predict the MESH ID for a TF-IDF vectorized phrase
        Note: A custom weight matrix can be substituted in; e.g. W=I is TF-IDF baseline.
        """
        W = self.W if W is None else W
        b = self.b if b is None else b

        s = x * W * self.D.T + b * x * self.D.T

        if s.nnz == 0:
            return None

        # Note: We convert to COO format here so as to to do an OM-more efficient
        # argmax over the raw data
        # TODO: Return set of tied top values -> pass to disambiguator!
        s = s.tocoo()
        if s.data.max() > 0.0:
            return self.cid_sets[s.col[s.data.argmax()]]
        else:
            return None


