import os, sys
from collections import defaultdict
import scipy.sparse as sparse
from .models import Candidate, Feature

# Feature modules
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree, XMLTree
from utils import get_as_dict
from entity_features import *


def load_all_features(candidate_set, session):
    """
    Given a CandidateSet and Session, generates (Candidate, <feature name>) pairs for all Candidates in the set.
    """
    for x in session.query(Candidate).filter(Candidate.set == candidate_set).join(Feature):
        yield x


class Featurizer(object):
    def __init__(self):
        self.feat_index = None
        self.inv_index  = None

    def get_feats(self, candidates):
        raise NotImplementedError()

    def fit_transform(self, candidates):
        """Assembles the set of features to be used, and applies this transformation to the candidates"""
        # Assemble and return the sparse feature matrix
        f_index = defaultdict(list)
        for f, i in self.get_feats(candidates):
            f_index[f].append(i)

        # Assemble and return sparse feature matrix
        # Also assemble reverse index of feature matrix index -> feature verbose name
        self.feat_index = {}
        self.inv_index  = {}
        F               = sparse.lil_matrix((len(candidates), len(f_index.keys())))
        for j,f in enumerate(f_index.keys()):
            self.feat_index[f] = j
            self.inv_index[j]  = f
            for i in f_index[f]:
                F[i,j] = 1
        return F

    def transform(self, candidates):
        """Given feature set has already been fit, simply apply to candidates."""
        F = sparse.lil_matrix((len(candidates), len(self.feat_index.keys())))
        for f,i in self.get_feats(candidates):
            if self.feat_index.has_key(f):
                F[i,self.feat_index[f]] = 1
        return F


class SpanPairFeaturizer(Featurizer):
    """Featurizer for SpanPair objects"""

    def __init__(self):
        self.feature_generator = compile_relation_feature_generator()
        super(SpanPairFeaturizer, self).__init__()

    def get_feats(self, span_pairs):
        for i,sp in enumerate(span_pairs):
            xmltree = corenlp_to_xmltree(get_as_dict(sp.span0.context))
            s1_idxs = range(sp.span0.get_word_start(), sp.span0.get_word_end() + 1)
            s2_idxs = range(sp.span1.get_word_start(), sp.span1.get_word_end() + 1)

            # Apply TDL features
            for f in self.feature_generator(xmltree.root, s1_idxs, s2_idxs):
                yield 'TDL_' + f, i


class SpanFeaturizer(Featurizer):
    """Featurizer for Span objects"""

    def __init__(self):
        self.feature_generator = compile_entity_feature_generator()
        super(SpanFeaturizer, self).__init__()

    def get_feats(self, spans):
        for i,s in enumerate(spans):
            sent    = get_as_dict(s.context)
            xmltree = corenlp_to_xmltree(sent)
            sidxs   = range(s.get_word_start(), s.get_word_end() + 1)

            # Add DDLIB entity features
            for f in get_ddlib_feats(sent, sidxs):
                yield 'DDL_' + f, i

            # Add TreeDLib entity features
            for f in self.feature_generator(xmltree.root, sidxs):
                yield 'TDL_' + f, i
