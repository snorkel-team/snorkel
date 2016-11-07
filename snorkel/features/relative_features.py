import os, sys

sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))

from .models import Span
from functools import partial
from string import punctuation
from tree_structs import corenlp_to_xmltree
from treedlib import compile_relation_feature_generator
from utils import get_as_dict


def get_span_splits(candidate, stopwords=None):
    split_pattern = r'[\s{}]+'.format(re.escape(punctuation))
    for i, arg in enumerate(candidate.get_arguments()):
        for token in re.split(split_pattern, s.get_span().lower()):
            if stopwords is None or token not in stopwords:
                yield 'SPAN_SPLIT[{0}][{1}]'.format(i, token), 1


def get_span_splits_stopwords(stopwords):
    return partial(get_span_splits, stopwords=stopwords)


def get_span_feats(candidate, stopwords=None):
    args = candidate.get_arguments()
    if not isinstance(args[0], Span):
        raise ValueError("Accepts Span-type arguments, %s-type= found.")

    # Unary candidates
    if len(args) == 1:
        get_tdl_feats = compile_entity_feature_generator()
        span          = args[0]
        sent          = get_as_dict(span.parent)
        xmltree       = corenlp_to_xmltree(sent)
        sidxs         = range(span.get_word_start(), span.get_word_end() + 1)
        if len(sidxs) > 0:

            # Add DDLIB entity features
            for f in get_ddlib_feats(sent, sidxs):
                yield 'DDL_' + f, 1

            # Add TreeDLib entity features
            for f in get_tdl_feats(xmltree.root, sidxs):
                yield 'TDL_' + f, 1

    # Binary candidates
    elif len(args) == 2:
        get_tdl_feats = compile_relation_feature_generator()
        span1, span2  = args
        xmltree       = corenlp_to_xmltree(get_as_dict(span1.parent))
        s1_idxs       = range(span1.get_word_start(), span1.get_word_end() + 1)
        s2_idxs       = range(span2.get_word_start(), span2.get_word_end() + 1)
        if len(s1_idxs) > 0 and len(s2_idxs) > 0:

            # Apply TDL features
            for f in get_tdl_feats(xmltree.root, s1_idxs, s2_idxs, stopwords=stopwords):
                yield 'TDL_' + f, 1
    else:
        raise NotImplementedError("Only handles unary and binary candidates")


def get_span_feats_stopwords(stopwords):
    return partial(get_span_feats, stopwords=stopwords)

