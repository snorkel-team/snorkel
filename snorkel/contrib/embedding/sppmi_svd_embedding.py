import gensim
import numpy as np

from collections import defaultdict
from utils import DEFAULT_STOPS, Embedder, strip_special
from scipy import sparse


class SnorkelSentenceGensimCorpus(gensim.interfaces.CorpusABC):

	def __init__(self, documents, tokens='words', stopwords=DEFAULT_STOPS,
				 subsample_t=1e-5, min_count=2):
		"""Converts Snorkel Documents to a Gensim corpus"""
		self.documents  = documents
		self.tokens     = tokens
		self.stopwords  = stopwords
		self.t          = subsample_t
		self.min_count  = min_count
		self.dictionary = gensim.corpora.dictionary.Dictionary(prune_at=None)
		print("Processing corpus with {0} documents".format(len(documents)))
		self._process_tokens()
		print("Corpus processing done!")

	def _filter(self, tokens, lower=True):
		"""Filter out stopwords and single-characters from a word sequence
		Optionally converts words to lowercase
		"""
		z = filter(lambda w: len(w) > 1 and w not in self.stopwords, tokens)
		if lower:
			return [w.lower() for w in map(strip_special, z)]
		return map(strip_special, z)

	def _token_seq_generator(self):
		"""Iterator over sentences producing their tokens"""
		for document in self.documents:
			for sentence in document.sentences:
				yield self._filter(sentence.__dict__[self.tokens])	

	def _process_tokens(self):
		"""Initialize dictionary and corpus token counts"""
		# Record absolute appearance counts for tokens
		counts = defaultdict(int)
		for sent_tokens in self._token_seq_generator():
			# Add to dictionary
			self.dictionary.doc2bow(sent_tokens, allow_update=True)
			# Update counts
			for token in sent_tokens:
				counts[token] += 1
		# Remove infrequent words
		print("\t{0} words in corpus".format(len(self.dictionary.token2id)))
		self.dictionary.filter_extremes(
			no_below=self.min_count, no_above=1.0, keep_n=None
		)
		# Dirty subsampling
		keys, s = self.dictionary.token2id.keys(), sum(counts.values())
		f = np.ravel([counts[k] for k in keys]) / float(s)
		p = 1.0 - np.sqrt(self.t / f)
		keep = (np.random.random(len(keys)) < p)
		bad_ids = [
			self.dictionary.token2id[k] for k, y in zip(keys, keep) if not y
		]
		self.dictionary.filter_tokens(bad_ids=bad_ids)
		print("\t{0} words after filter".format(len(self.dictionary.token2id)))

	def iter_sentences(self):
		for sent_tokens in self._token_seq_generator():
			yield [
				self.dictionary.token2id[token] for token in sent_tokens
				if token in self.dictionary.token2id
			]

	def __iter__(self):
		for sent_tokens in self._token_seq_generator():
			yield self.dictionary.doc2bow(sent_tokens, allow_update=False)

	def __len__(self):
		return len(self.documents)


class SPPMISVDEmbedder(Embedder):

	def __init__(self, corpus, window_size=2):
		"""Embed words and sentences based on SVD of the SPPMI matrix
			@corpus: Gensim-style corpus (see SnorkelSentenceGensimCorpus)

		Implementation choices are inspired by
			http://www.aclweb.org/anthology/Q15-1016
		"""
		super(SPPMISVDEmbedder, self).__init__(corpus, None)
		self.dictionary = corpus.dictionary
		self.w          = window_size
		print("Processing corpus with context window size 2")
		self.D, self.W, self.C, self.token_ct = self._process_corpus()
		print("Corpus processing done!")

	def _process_corpus(self):
		# Construct co-occurence and count matrics
		D, W, C = defaultdict(int), defaultdict(int), defaultdict(int)
		ct = defaultdict(int)
		for sent in self.corpus.iter_sentences():
			for c, word in enumerate(sent):
				ct[word] += 1
				# Iterate over context window
				for j in range(max(0, c-self.w), min(len(sent), c+self.w+1)):
					if j == c:
						continue
					W[ word          ] += 1
					C[       sent[j] ] += 1
					D[(word, sent[j])] += 1
		return D, W, C, ct
		
	def run_sppmi_svd(self, rank=50, alpha_cds=0.75, k=1):
		"""Generate embeddings from SPPMI matrix
			@rank:      dimensionality of the word embeddings
			@alpha_cds: alpha parameter for context distribution smoothing
			@k:         log shift parameter (number of negative samples)
		"""
		# Context distribution smoothing
		C_cds = {k: v**alpha_cds for k, v in self.C.iteritems()}
		t_cds = float(sum(C_cds.values()))
		# Construct SPPMI matrix
		M = sparse.lil_matrix((len(self.dictionary), len(self.dictionary)))
		neg = np.log(k)
		for (w, c), v in self.D.iteritems():
			# Be safe about assigning zeros to sparse matrix
			spmi = np.log((v * t_cds) / (self.W[w] * C_cds[c])) - neg
			if spmi > 0:
				M[w, c] = spmi
		# Sparse SVD
		self.U, self.S, _ = sparse.linalg.svds(
			M, k=rank, tol=1e-12, return_singular_vectors='u'
		)

	def word_embeddings(self, p=0.5):
		return np.dot(self.U, np.diag(np.power(self.S, p)))
