import os, sys
from collections import defaultdict
import numpy as np
import scipy.sparse as sparse
import itertools

# Feature modules
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree, XMLTree
from entity_features import *


class Featurizer(object):
    """
    A Featurizer applies a set of **feature generators** to each Candidate,
    based on (i) the arity of the candidate, and (ii) the _associated Contexts_.
    
    The apply() function takes in N candidates, and returns an N x F sparse matrix,
    where F is the dimension of the feature space.
    """
    def __init__(self, arity=1):
        self.arity          = arity
        self.feat_index     = None
        self.feat_inv_index = None
    
    def _generate_context_feats(self, get_feats, prefix, candidates):
        """
        Given a function that given a candidate, generates features, _using a specific context_,
        and a unique prefix string for this context, return a generator over features (as strings).
        """
        for i,c in enumerate(candidates):
            for f in get_feats(c):
                yield i, prefix + f

    def _match_contexts(self, candidates):
        """Given the candidates, and using _generate_context_feats, return a list of generators."""
        raise NotImplementedError()

    def apply(self, candidates):
        """Given feature set has already been fit, simply apply to candidates."""
        F                  = sparse.lil_matrix((len(candidates), len(self.feat_index.keys())))
        feature_generators = self._match_contexts(candidates)
        for i,f in itertools.chain(*feature_generators):
            F[i,self.feat_index[f]] = 1
        return F

    def fit_apply(self, candidates):
        """Assembles the set of features to be used, and applies this transformation to the candidates"""
        feature_generators = self._match_contexts(candidates)

        # Assemble and return the sparse feature matrix
        f_index = defaultdict(list)
        for i,f in itertools.chain(*feature_generators):
            f_index[f].append(i)

        # Assemble and return sparse feature matrix
        # Also assemble reverse index of feature matrix index -> feature verbose name
        self.feat_index     = {}
        self.feat_inv_index = {}
        F                   = sparse.lil_matrix((len(candidates), len(f_index.keys())))
        for j,f in enumerate(f_index.keys()):
            self.feat_index[f] = j
            self.feat_inv_index[j] = f
            for i in f_index[f]:
                F[i,j] = 1
        return F


class NgramFeaturizer(Featurizer):
    """Feature for relations (of arity >= 1) defined over Ngram objects."""
    def _match_contexts(self, candidates):
        feature_generators = []

        # Unary relations
        if self.arity == 1:
            cidxs = range(c.word_start, c.word_end+1)

            # Add DDLIB entity features
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_ddlib_feats(c, cidxs), 'DDLIB_', candidates))

            # Add TreeDLib entity features
            if candidates[0].xmltree is not None:
                get_feats = compile_entity_feature_generator()
                feature_generators.append(self._generate_context_feats( \
                    lambda c : get_feats(c.xmltree.root, cidxs), 'TDLIB_', candidates))

        if self.arity == 2:
            c1idxs = range(c.e1.word_start, c.e1.word_end+1)
            c2idxs = range(c.e2.word_start, c.e2.word_end+1)

            # Add TreeDLib relation features
            if candidates[0].xmltree.root is not None:
                get_feats = compile_relation_feature_generator()
                feature_generators.append(self._generate_context_feats( \
                    lambda c : get_feats(c.xmltree.root, c1idxs, c2idxs), 'TDLIB_', candidates))
        return feature_generators


class LegacyCandidateFeaturizer(Featurizer):
    """Temporary class to handle v0.2 Candidate objects."""
    def _match_contexts(self, candidates):
        feature_generators = []

        # Unary relations
        if self.arity == 1:

            # Add DDLIB entity features
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_ddlib_feats(c, c.idxs), 'DDLIB_', candidates))

            # Add TreeDLib entity features
            if candidates[0].root is not None:
                get_feats = compile_entity_feature_generator()
                feature_generators.append(self._generate_context_feats( \
                    lambda c : get_feats(c.root, c.idxs), 'TDLIB_', candidates))

        if self.arity == 2:

            # Add TreeDLib relation features
            if candidates[0].root is not None:
                get_feats = compile_relation_feature_generator()
                feature_generators.append(self._generate_context_feats( \
                    lambda c : get_feats(c.root, c.e1_idxs, c.e2_idxs), 'TDLIB_', candidates))
        return feature_generators
    
