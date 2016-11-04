import os, sys
import numpy as np
from collections import defaultdict
import scipy.sparse as sparse
from .models import Candidate, CandidateSet, Feature, TemporarySpan, Phrase
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree
from utils import get_as_dict
from entity_features import *


def get_span_feats(candidate):
    args = candidate.get_arguments()
    if not (isinstance(args[0], TemporarySpan)):
        raise ValueError("Accepts Span-type arguments, %s-type found." % type(candidate))

    # Unary candidates
    if len(args) == 1:
        get_tdl_feats = compile_entity_feature_generator()
        get_tablelib_feats = tablelib_unary_features
        get_vizlib_feats = vizlib_unary_features
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

            # Add TableLib entity features (if applicable)
            if isinstance(span.parent, Phrase):
                for f in get_tablelib_feats(span):
                    yield 'TAB_' + f, 1
                for f in get_vizlib_feats(span):
                    yield 'VIZ_' + f, 1

    # Binary candidates
    elif len(args) == 2:
        get_tdl_feats = compile_relation_feature_generator()
        get_tablelib_feats = tablelib_binary_features
        get_vizlib_feats = vizlib_binary_features
        span1, span2  = args
        xmltree       = corenlp_to_xmltree(get_as_dict(span1.parent))
        s1_idxs       = range(span1.get_word_start(), span1.get_word_end() + 1)
        s2_idxs       = range(span2.get_word_start(), span2.get_word_end() + 1)
        if len(s1_idxs) > 0 and len(s2_idxs) > 0:

            # Apply TreeDLib relation features
            for f in get_tdl_feats(xmltree.root, s1_idxs, s2_idxs):
                yield 'TDL_' + f, 1
            
            # TODO: add DDLib features for binary relations?

            # Add TableLib relation features (if applicable)
            if isinstance(span1.parent, Phrase) or isinstance(span2.parent, Phrase):
                for f in get_tablelib_feats(span1, span2):
                    yield 'TAB_' + f, 1
            if span1.has_visuals() or span2.has_visuals():
                for f in get_vizlib_feats(span1, span2):
                    yield 'VIZ_' + f, 1
    else:
        raise NotImplementedError("Only handles unary and binary candidates currently")
