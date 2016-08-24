"""Classes for extracting features from candidates.

Feature extractors are subclasses of Featurizer. They implement the featurize
method, which return a generator of features from a set of candidates.

To write a custom featurizer, one need sto subclass Featurize and implement
the featurize() method.
"""

import os, sys
from collections import defaultdict
import numpy as np
import scipy.sparse as sparse
import itertools

# Feature modules
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree, XMLTree
from utils import get_as_dict
from entity_features import get_table_feats, get_relation_table_feats, \
    get_ddlib_feats, compile_entity_feature_generator


class Featurizer(object):
    """
    A Featurizer applies a set of features to each Candidate,
    based on (i) the arity of the candidate, and (ii) the _associated Contexts_.

    The transform() function takes in N candidates, and returns an N x F sparse matrix,
    where F is the dimension of the feature space.
    """
    def __init__(self, arity=1):
        self.arity          = arity
        self.feat_index     = None
        self.feat_inv_index = None

    def featurize(self, candidates):
        """Given the candidates, return a generator over features"""
        raise NotImplementedError()

    def transform(self, candidates):
        """Given feature set has already been fit, simply apply to candidates."""
        F                 = sparse.lil_matrix((len(candidates), len(self.feat_index.keys())))
        feature_generator = self.featurize(candidates)
        for i,f in feature_generator: # itertools.chain(*feature_generators):
            if self.feat_index.has_key(f):
                F[i,self.feat_index[f]] = 1
        return F

    def fit_transform(self, candidates):
        """Assembles the set of features to be used, and applies this transformation to the candidates"""
        feature_generator = self.featurize(candidates)

        # Assemble and return the sparse feature matrix
        f_index = defaultdict(list)
        print 'Building feature index...'
        for i,f in feature_generator: # itertools.chain(*feature_generators):
            f_index[f].append(i)

        # Assemble and return sparse feature matrix
        # Also assemble reverse index of feature matrix index -> feature verbose name
        self.feat_index     = {}
        self.feat_inv_index = {}
        F                   = sparse.lil_matrix((len(candidates), len(f_index.keys())))
        n_tot = len(f_index.keys())
        print 'Extracting features...'
        for j,f in enumerate(f_index.keys()):
            if j % 5000 == 0: print '%d/%d' % (j, n_tot)
            self.feat_index[f] = j
            self.feat_inv_index[j] = f
            for i in f_index[f]:
                F[i,j] = 1
        return F

    def get_features_by_candidate(self, candidate):
        feature_generator = self.featurize([candidate])
        feats = []
        for i,f in feature_generator: # itertools.chain(*feature_generators):
            feats.append(f)
        return feats

class NgramFeaturizer(Featurizer):
    """Feature for relations (of arity >= 1) defined over Ngram objects."""

    def featurize(self, candidates):
        feature_generators = []

        # Unary relations
        if self.arity == 1:

            # Add DDLIB entity features
            feature_generators.append(_generate_context_feats( \
                lambda c : get_ddlib_feats(c, range(c.get_word_start(), c.get_word_end()+1)), 'DDLIB_', candidates))

            # Add TreeDLib entity features
            # get_feats = compile_entity_feature_generator()
            # feature_generators.append(_generate_context_feats( \
            #     lambda c : get_feats(c.context['xmltree'].root, range(c.word_start, c.word_end+1)), 'TDL_', candidates))

        if self.arity == 2:
            raise NotImplementedError("Featurizer needs to be implemented for binary relations!")

        return itertools.chain(*feature_generators)


class TableNgramFeaturizer(NgramFeaturizer):
    """In addition to Ngram features, add table structure information."""
    def featurize(self, candidates):
        feature_generators = [super(TableNgramFeaturizer, self).featurize(candidates)]

        # Unary relations
        if self.arity == 1:
            feature_generators.append(_generate_context_feats( \
                lambda c : get_table_feats(c), 'TABLE_', candidates))

        if self.arity == 2:
            raise NotImplementedError("Featurizer needs to be implemented for binary relations!")

        return itertools.chain(*feature_generators)

class TableNgramPairFeaturizer(TableNgramFeaturizer):
    def _prepend_entity_label(self, generator, entity_label):
        for feat in generator:
            yield (feat[0], ('e%s_' % entity_label) + feat[1])

    def featurize(self, candidates):
        # TODO: generalize this to arity=N
        # collect (entity) feature generators from parent
        e0_feature_generator = TableNgramFeaturizer.featurize(
            self, [candidate.span0 for candidate in candidates])
        e1_feature_generator = TableNgramFeaturizer.featurize(
            self, [candidate.span1 for candidate in candidates])

        feature_generators = [
            self._prepend_entity_label(e0_feature_generator,0),
            self._prepend_entity_label(e1_feature_generator,1)
        ]

        feature_generators.append(_generate_context_feats( \
            lambda c : get_relation_table_feats(c), 'RTABLE_', candidates))

        return itertools.chain(*feature_generators)

class NgramPairFeaturizer(NgramFeaturizer):
    def _prepend_entity_label(self, generator, entity_label):
        for feat in generator:
            yield (feat[0], ('e%s_' % entity_label) + feat[1])

    def featurize(self, candidates):
        # TODO: generalize this to arity=N
        # collect (entity) feature generators from parent
        e0_feature_generator = NgramFeaturizer.featurize(
            self, [candidate.span0 for candidate in candidates])
        e1_feature_generator = NgramFeaturizer.featurize(
            self, [candidate.span1 for candidate in candidates])

        feature_generators = [
            self._prepend_entity_label(e0_feature_generator,0),
            self._prepend_entity_label(e1_feature_generator,1)
        ]

        # TODO: add features derived from pair (e.g. distance in sentence)

        return itertools.chain(*feature_generators)

class UnionFeaturizer(Featurizer):
    """Combines multiple featurizers into one"""

    def __init__(self, featurizer_list, arity=1):
        super(UnionFeaturizer, self).__init__(arity)
        self.featurizer_list = featurizer_list

    def featurize(self, candidates):
        feature_generators = []
        for featurizer in featurizer_list:
            feature_generators.append(featurizer.featurize(candidates))

        return itertools.chain(*feature_generators)

# ----------------------------------------------------------------------------
# helpers

def _generate_context_feats(get_feats, prefix, candidates):
    """
    Given a function that given a candidate, generates features, _using a specific context_,
    and a unique prefix string for this context, return a generator over features (as strings).
    """
    for i,c in enumerate(candidates):
        for f in get_feats(c):
            yield i, prefix + f