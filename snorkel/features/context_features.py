from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from collections import defaultdict
from functools import partial
from snorkel.models import Span


def get_token_count_feats(candidate, context, attr, ngram, stopwords):
    """Base function for n-gram count features
    candidate: @Candidate to extract features for
    context: @Context over which to count n-grams
    attr: @Sentence attribute to retrieve n-grams
    ngram: maximum n-gram length
    stopwords: @set of stopwords to filter out from counts
    """
    args = candidate.get_contexts()
    if not isinstance(args[0], Span):
        raise ValueError("Accepts Span-type arguments, %s-type found.")

    counter = defaultdict(int)
    # Count n-gram instances
    for tokens in (sent[attr] for sent in context.get_sentence_generator()):
        for i in range(len(tokens)):
            for j in range(i+1, min(len(tokens), i + ngram) + 1):
                counter[' '.join(tokens[i:j])] += 1
    # Yield counts if n-gram is not in stopwords
    for gram in counter:
        if (not stopwords) or not all([t in stopwords for t in gram.split()]):
            yield 'TOKEN_FEATS[' + gram + ']', counter[gram]


def get_document_token_count_feats_base(candidate, attr, ngram, stopwords):
    """Apply @get_token_count_feats over the parent @Document of @candidate"""
    doc = candidate.get_parent().get_parent()
    return get_token_count_feats(candidate, doc, attr, ngram, stopwords)


def get_sentence_token_count_feats_base(candidate, attr, ngram, stopwords):
    """Apply @get_token_count_feats over the parent @Sentence of @candidate"""
    sentence = candidate.get_parent().get_parent()
    return get_token_count_feats(candidate, sentence, attr, ngram, stopwords)


def get_document_token_count_feats(stopwords=None, ngram=1, attr='lemmas'):
    """Get a document token count unary function"""
    return partial(get_document_token_count_feats_base, attr=attr, ngram=ngram,
                   stopwords=stopwords)


def get_sentence_token_count_feats(stopwords=None, ngram=1, attr='lemmas'):
    """Get a sentence token count unary function"""
    return partial(get_sentence_token_count_feats_base, attr=attr, ngram=ngram,
                   stopwords=stopwords)
