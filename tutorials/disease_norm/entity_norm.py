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
    """
    Given:
        * X, an N x V phrase matrix
        * y, a corresponding N-length list of gold CIDs
        * D, an M x V dictionary term matrix
        * d, a corresponding M-length list of CIDs
    Run SGD to learn the matrix of weights W

    Just initialize with (D, d); then use self.train(X,y)
    """
    def __init__(self, D, d):
        self.D = D
        self.d = d
        
        self.cids = frozenset(d)
        self.V    = D.shape[1]
        
        # Index the dictionary terms (D) by cid (d)
        self.cid_to_rows = defaultdict(list)
        for i, cid in enumerate(self.d):
            self.cid_to_rows[cid].append(i)
    
    def train(self, X, y, rate=1e-3, n_iter=3, n_iter_neg=10, sample_close_negs=True, verbose=True):
        """
        If sample_close_negs = True, negative terms will be sampled only from "close"
        terms to the phrase, i.e. with _some_ direct word overlap.
        """
        N = X.shape[0]

        # Build overlap matrix
        if sample_close_negs:
            print "Building close negatives dictionary..."
            M          = X * self.D.T
            close_negs = {}
            for i in range(N):
                close_negs[i] = set([self.d[j] for j in M.getrow(i).nonzero()[1]]).difference([y[i]])

        # Initialize W = I
        W = identity(self.V, format='csr')

        # Cosine similarity term
        b = 1.0

        # Run SGD
        for it in range(n_iter):
            sys.stdout.write("\rIteration: %s" % it)

            # Iterate in random order through the training examples
            run = range(N)
            random.shuffle(run)
            for i in run:

                # Randomly pick a tuple (x, t^+, t^-)
                # First pick x, a random training phrase
                x = X.getrow(i)

                # NOTE: Skip OOD training examples!
                cid = y[i]
                if cid not in self.cid_to_rows:
                    continue

                # Next pick tp, a random dictionary term which maps to the correct CID for x
                j  = random.choice(self.cid_to_rows[cid])
                tp = self.D.getrow(j)

                # Next pick tp, a random dictionary term which maps to an *incorrect* CID for x
                # Take multiple samples for this training example
                neg_cids = close_negs[i] if sample_close_negs and len(close_negs[i]) > 0 else self.cids.difference([cid])
                for itn in range(n_iter_neg):
                    cid_neg  = random.sample(neg_cids, 1)[0]
                    k        = random.choice(self.cid_to_rows[cid_neg])
                    tn       = self.D.getrow(k)

                    # Take gradient step
                    if 1 - (x * W * tp.T)[0,0] - b*(x * tp.T)[0,0] + (x * W * tn.T)[0,0] + b*(x * tn.T)[0,0] > 0:
                        W = W + rate * ( x.T * tp - x.T * tn )
                        b = b + rate * ( (x * tp.T)[0,0] - (x * tn.T)[0,0])
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

        s = x * W * self.D.T + self.b * x * self.D.T

        if s.nnz == 0:
            return None

        # Note: We convert to COO format here so as to to do an OM-more efficient
        # argmax over the raw data
        s = s.tocoo()
        if s.data.max() > 0.0:
            return self.d[s.col[s.data.argmax()]]
        else:
            return None


