import gensim

from collections import defaultdict
from string import punctuation


class SnorkelGensimCorpus(gensim.interfaces.CorpusABC):

	def __init__(self, documents, tokens='words', stopwords=set(punctuation)):
		"""Converts Snorkel Documents to a Gensim corpus"""
		self.documents  = documents
		self.tokens     = tokens
		self.stopwords  = stopwords
		self.dictionary = gensim.corpora.dictionary.Dictionary()
		self.token_ct   = defaultdict(int)
		self._process_tokens()

	def _filter(self, tokens):
		"""Filter out stopwords from a word sequence"""
		return filter(lambda w: w not in self.stopwords, tokens)

	def _token_seq_generator(self):
		"""Iterator over documents producing iterators over their tokens"""
		for document in self.documents:
			yield (
				tok for sentence in document.sentences 
				for tok in self._filter(sentence.__asdict__[self.tokens])
			)

	def _process_tokens(self):
		"""Initialize dictionary and corpus token counts"""
		# Record absolute appearance counts for tokens
		counts = defaultdict(int)
		for doc_tokens in self._token_seq_generator():
			# Add to dictionary
			self.dictionary.doc2bow(doc_tokens, allow_update=True)
			# Update counts
			for token in doc_tokens:
				counts[token] += 1
		# Filter extremes (TODO: make parameters accessible)
		self.dictionary.filter_extremes(no_below=2, no_above=0.9, keep_n=None)
		# Replace count dictionary keys with tokenIDs
		self.token_ct = defaultdict(int)
		for token, ct in counts.iteritems():
			self.token_ct[self.dictionary.token2id[token]] = ct

	def __iter__(self):
		for doc_tokens in self._token_seq_generator():
			yield self.dictionary.doc2bow(doc_tokens, allow_update=False)

	def __len__(self):
		return len(self.documents)

		


class LSAEmbedder(object):

	def __init__(self, corpus):
		"""Embed words and sentences based on latent semantic analysis
			@corpus: Gensim-style corpus (see SnorkelGensimCorpus)

		LSA is run on @corpus to get word embeddings
		Sentence embeddings are computed by an implementation of
			https://openreview.net/pdf?id=SyK00v5xx
		"""
		self.corpus     = corpus
		self.dictionary = corpus.dictionary
		self.token_ct   = token_ct
		self.fname      = 'lsa_snorkel'
		self.tfidf_mm   = None
		self.lsa        = None
		self._process_corpus()

	def _process_corpus(self):
		# Get MatrixMarket format corpus
		gensim.corpora.MmCorpus.serialize(
			self.fname + '.mm', self.corpus, progress_cnt=100
		)
		mm_corpus = gensim.corpora.MmCorpus(self.fname + '.mm')
		# Get TF-IDF model
		tfidf = gensim.models.TfidfModel(
			mm_corpus, id2word=self.dictionary, normalize=True
		)
		gensim.corpora.MmCorpus.serialize(
			self.fname + '_tfidf.mm', tfidf[mm_corpus], progress_cnt=100
		)
		# Reload as Matrix Market format
		self.tfidf_mm = gensim.corpora.MmCorpus(self.fname + '_tfidf.mm')

	def run_lsa(self, n_topics=200):
		"""Run latent semantic analysis"""
		self.lsa = lsi = gensim.models.lsimodel.LsiModel(
			corpus=self.tfidf_mm, id2word=self.dictionary, num_topics=n_topics
		)

	def marginal_estimates(self):
		s = sum(self.token_ct.values())
		return {k: float(v) / s for k, v in self.token_ct.iteritems()}
