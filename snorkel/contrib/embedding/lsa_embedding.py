import gensim
import numpy as np

from collections import defaultdict
from utils import DEFAULT_STOPS, Embedder, strip_special


class SnorkelGensimCorpus(gensim.interfaces.CorpusABC):

	def __init__(self, documents, tokens='words', stopwords=DEFAULT_STOPS,
				 bigrams=False):
		"""Converts Snorkel Documents to a Gensim corpus"""
		self.documents  = documents
		self.tokens     = tokens
		self.stopwords  = stopwords
		self.dictionary = gensim.corpora.dictionary.Dictionary()
		self.token_ct   = defaultdict(int)
		self.bigrams    = bigrams
		self._process_tokens()

	def _gen_grams(self, tokens):
		"""Generate unigrams or bigrams from a unigram sequence"""
		if not self.bigrams:
			return tokens
		grams = list(tokens)
		for i in xrange(len(tokens) - 1):
			grams.append('{0} {1}'.format(grams[i], grams[i+1]))
		return grams

	def _filter(self, tokens):
		"""Filter out stopwords and single-characters from a word sequence"""
		z = filter(lambda w: len(w) > 1 and w not in self.stopwords, tokens)
		return [strip_special(w) for w in z]

	def _token_seq_generator(self):
		"""Iterator over documents producing iterators over their tokens"""
		for document in self.documents:
			yield [
				tok.lower() for sentence in document.sentences for tok in
				self._gen_grams(self._filter(sentence.__dict__[self.tokens]))
			]		

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
			if token in self.dictionary.token2id:
				self.token_ct[self.dictionary.token2id[token]] = ct

	def iter_sentences(self):
		for document in self.documents:
			for sentence in document.sentences:
				sent_tokens = (
					tok.lower() for tok in 
					self._filter(sentence.__dict__[self.tokens])
				)
				yield [
					self.dictionary.token2id[token] for token in sent_tokens
					if token in self.dictionary.token2id
				]


	def iter_documents(self):
		for doc_tokens in self._token_seq_generator():
			yield [
				self.dictionary.token2id[token] for token in doc_tokens
				if token in self.dictionary.token2id
			]

	def __iter__(self):
		for doc_tokens in self._token_seq_generator():
			yield self.dictionary.doc2bow(doc_tokens, allow_update=False)

	def __len__(self):
		return len(self.documents)


class LSAEmbedder(Embedder):

	def __init__(self, corpus):
		"""Embed words and sentences based on latent semantic analysis
			@corpus: Gensim-style corpus (see SnorkelGensimCorpus)
		LSA is run on @corpus to get word embeddings
		"""
		super(LSAEmbedder, self).__init__(corpus, corpus.token_ct)
		self.dictionary = corpus.dictionary
		self.fname      = 'lsa_snorkel'
		self.tfidf_mm   = None
		self.lsa        = None
		print("Processing corpus")
		self._process_corpus()
		print("Corpus processed!")

	def _process_corpus(self):
		# Get MatrixMarket format corpus
		print("\tConverting corpus")
		gensim.corpora.MmCorpus.serialize(
			self.fname + '.mm', self.corpus, progress_cnt=100
		)
		mm_corpus = gensim.corpora.MmCorpus(self.fname + '.mm')
		# Get TF-IDF model
		print("\tComputing TF-IDF")
		tfidf = gensim.models.TfidfModel(
			mm_corpus, id2word=self.dictionary, normalize=True
		)
		gensim.corpora.MmCorpus.serialize(
			self.fname + '_tfidf.mm', tfidf[mm_corpus], progress_cnt=100
		)
		# Reload as Matrix Market format
		print("\tConverting TF-IDF")
		self.tfidf_mm = gensim.corpora.MmCorpus(self.fname + '_tfidf.mm')

	def run_lsa(self, n_topics=200):
		"""Run latent semantic analysis"""
		self.lsa = lsi = gensim.models.lsimodel.LsiModel(
			corpus=self.tfidf_mm, id2word=self.dictionary, num_topics=n_topics
		)

	def word_embeddings(self, scaled=False):
		if scaled:
			return (1.0 / self.lsa.projection.s) * self.lsa.projection.u
		return self.lsa.projection.u
