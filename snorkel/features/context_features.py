from .models import Span
from collections import defaultdict
from functools import partial


def get_token_count_feats(candidate, token_generator, attr='lemmas',
                          ngram=1, stopwords=None):
    args = candidate.get_arguments()
    if not isinstance(args[0], Span):
        raise ValueError("Accepts Span-type arguments, %s-type found.")

    counter = defaultdict(int)

    for tokens in token_generator(candidate, attr):
        for i in xrange(len(tokens)):
            for j in range(i+1, min(len(tokens), i + ngram) + 1):
                counter[' '.join(tokens[i:j])] += 1

    for gram in counter:
        if (not stopwords) or not all([t in stopwords for t in gram.split()]): 
            yield 'TOKEN_FEATS[' + gram + ']', counter[gram]


def sentence_token_generator(candidate, attr):
    return (sent[attr] for sent in candidate.get_parent().get_sentence_generator())


def doc_token_generator(candidate, attr):
    return (sent[attr] for sent in candidate.get_parent().parent.get_sentence_generator())


def get_sentence_count_feats(stopwords, ngram=1, attr='lemmas'):
    return partial(get_token_count_feats, token_generator=sentence_token_generator,
        attr=attr, ngram=ngram, stopwords=stopwords)


def get_document_count_feats(stopwords, ngram=1, attr='lemmas'):
    return partial(get_token_count_feats, token_generator=doc_token_generator,
        attr=attr, ngram=ngram, stopwords=stopwords)
