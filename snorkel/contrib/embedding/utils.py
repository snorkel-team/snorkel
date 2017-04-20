import numpy as np

from sklearn.decomposition import PCA
from string import punctuation

# Constants
STOPWORDS = set([
	'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
	'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
	'will', 'with',
])
DEFAULT_STOPS = STOPWORDS | set(punctuation)

# Util functions
def strip_special(w):
	"""Remove weird characters from strings"""
	return ''.join(c for c in w if ord(c) < 128)

# Generic class
class Embedder(object):

	def __init__(self, corpus, token_ct):
		"""Embedder generic class
		Sentence embeddings are computed by an implementation of
			https://openreview.net/pdf?id=SyK00v5xx
		"""
		self.corpus   = corpus
		self.token_ct = token_ct

	def word_embeddings(self, **kwargs):
		raise NotImplementedError()

	def marginal_estimates(self):
		s = sum(self.token_ct.values())
		marginals = np.zeros(len(self.token_ct))
		for k, v in self.token_ct.iteritems():
			marginals[k] = float(v) / s
		return marginals

	def embed_sentences(self, a=1e-2, **embedding_kwargs):
		X = []
		# Get word embeddings and corpus marginals
		U = self.word_embeddings(**embedding_kwargs)
		p = self.marginal_estimates()
		for sentence in self.corpus.iter_sentences():
			# Get token indices
			w = np.ravel(sentence)
			if len(w) == 0:
				X.append(np.zeros(U.shape[1]))
				continue
			# Normalizer
			z = 1.0 / len(w)
			# Embed sentence
			q = np.sum((a / (a + p[w])).reshape((len(w), 1)) * U[w, :], axis=0)
			X.append(z * q)
		# Compute first principal component
		X = np.array(X)
		pca = PCA(n_components=1, whiten=False, svd_solver='randomized')
		pca.fit(X)
		K = np.dot(pca.components_.T, pca.components_)
		return X - np.dot(X, K)
