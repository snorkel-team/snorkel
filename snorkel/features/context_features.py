from .models import Span
from collections import defaultdict
from functools import partial


def get_token_count_feats(candidate, context, attr, ngram, stopwords):
    args = candidate.get_arguments()
    if not isinstance(args[0], Span):
        raise ValueError("Accepts Span-type arguments, %s-type found.")

    counter = defaultdict(int)

    for tokens in (sent[attr] for sent in context.get_sentence_generator()):
        for i in xrange(len(tokens)):
            for j in range(i+1, min(len(tokens), i + ngram) + 1):
                counter[' '.join(tokens[i:j])] += 1

    for gram in counter:
        if (not stopwords) or not all([t in stopwords for t in gram.split()]):
            yield 'TOKEN_FEATS[' + gram + ']', counter[gram]


def get_document_token_count_feats_base(candidate, attr, ngram, stopwords):
    doc = candidate.get_parent().parent
    return get_token_count_feats(candidate, doc, attr, ngram, stopwords)


def get_sentence_token_count_feats_base(candidate, attr, ngram, stopwords):
    sentence = candidate.get_parent().parent
    return get_token_count_feats(candidate, sentence, attr, ngram, stopwords)


def get_document_token_count_feats(stopwords=None, ngram=1, attr='lemmas'):
    return partial(get_document_token_count_feats_base, attr=attr, ngram=ngram,
                   stopwords=stopwords)


def get_sentence_token_count_feats(stopwords=None, ngram=1, attr='lemmas'):
    return partial(get_sentence_token_count_feats_base, attr=attr, ngram=ngram,
                   stopwords=stopwords)
