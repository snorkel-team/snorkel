import os, sys
import numpy as np
from pandas import DataFrame
from collections import defaultdict
import scipy.sparse as sparse
from .models import Candidate, CandidateSet, Feature, Span
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree
from utils import get_as_dict
from entity_features import *
from functools import partial


def get_span_feats(candidate):
    args = candidate.get_arguments()
    if not isinstance(args[0], Span):
        raise ValueError("Accepts Span-type arguments, %s-type found.")

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
            for f in get_tdl_feats(xmltree.root, s1_idxs, s2_idxs):
                yield 'TDL_' + f, 1
    else:
        raise NotImplementedError("Only handles unary and binary candidates currently")

def get_doc_feats(candidate, stopwords=None):
    args = candidate.get_arguments()
    if not isinstance(args[0], Span):
        raise ValueError("Accepts Span-type arguments, %s-type found.")

    counter = defaultdict(int)

    parent_doc_sentences = candidate[0].parent.document.sentences
    for sent in parent_doc_sentences:
        for lemma in sent.lemmas:
            counter[lemma] += 1

    for lemma in counter:
        if (not stopwords) or (lemma not in stopwords): 
            yield 'DOCFEATS_LEMMA[' + lemma + ']', counter[lemma]

def get_doc_feats_generator(stopwords):
    return partial(get_doc_feats, stopwords=stopwords)

def feats_from_matrix(candidate, candidate_index, X, prefix):
    i = candidate_index[candidate.id]
    for j in xrange(X.shape[1]):
        yield '{0}_{1}'.format(prefix, j), X[i, j]

def feats_from_matrix_generator(candidate_index, X, prefix='col'):
    return partial(feats_from_matrix, candidate_index=candidate_index, X=X, prefix=prefix)