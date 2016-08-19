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
    A Featurizer applies a set of **feature generators** to each Candidate,
    based on (i) the arity of the candidate, and (ii) the _associated Contexts_.

    The transform() function takes in N candidates, and returns an N x F sparse matrix,
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

    # TODO: Take this out...
    def _preprocess_candidates(self, candidates):
        return candidates

    def _match_contexts(self, candidates):
        """Given the candidates, and using _generate_context_feats, return a list of generators."""
        raise NotImplementedError()

    def transform(self, candidates):
        """Given feature set has already been fit, simply apply to candidates."""
        F                  = sparse.lil_matrix((len(candidates), len(self.feat_index.keys())))
        feature_generators = self._match_contexts(self._preprocess_candidates(candidates))
        for i,f in itertools.chain(*feature_generators):
            if self.feat_index.has_key(f):
                F[i,self.feat_index[f]] = 1
        return F

    def fit_transform(self, candidates):
        """Assembles the set of features to be used, and applies this transformation to the candidates"""
        feature_generators = self._match_contexts(self._preprocess_candidates(candidates))

        # Assemble and return the sparse feature matrix
        f_index = defaultdict(list)
        n_tot = len(candidates)
        first = True
        print 'Building feature index...'
        print '%d feature generators in total.' % len(feature_generators)
        for i,f in itertools.chain(*feature_generators):
            if i % 10000 == 0 and first: 
                print '%d/%d' % (i, n_tot)
                first = False
            if i % 10000 != 0: first = True
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
        feature_generators = self._match_contexts(self._preprocess_candidates([candidate]))
        feats = []
        for i,f in itertools.chain(*feature_generators):
            feats.append(f)
        return feats

class NgramFeaturizer(Featurizer):
    """Feature for relations (of arity >= 1) defined over Ngram objects."""
    def _preprocess_candidates(self, candidates):
        # for c in candidates:
            # if not isinstance(c.context, dict):
            #     c.context = get_as_dict(c.context)
            # if c.context['xmltree'] is None:
            #     c.context['xmltree'] = corenlp_to_xmltree(c.context)
        return candidates

    def _match_contexts(self, candidates):
        feature_generators = []

        # Unary relations
        if self.arity == 1:

            # Add DDLIB entity features
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_ddlib_feats(c, range(c.get_word_start(), c.get_word_end()+1)), 'DDLIB_', candidates))

            # Add TreeDLib entity features
            # get_feats = compile_entity_feature_generator()
            # feature_generators.append(self._generate_context_feats( \
            #     lambda c : get_feats(c.context['xmltree'].root, range(c.word_start, c.word_end+1)), 'TDL_', candidates))

        if self.arity == 2:
            raise NotImplementedError("Featurizer needs to be implemented for binary relations!")

        return feature_generators


class TableNgramFeaturizer(NgramFeaturizer):
    """In addition to Ngram features, add table structure information."""
    def _match_contexts(self, candidates):
        feature_generators = super(TableNgramFeaturizer, self)._match_contexts(candidates)

        # Unary relations
        if self.arity == 1:
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_table_feats(c), 'TABLE_', candidates))

        if self.arity == 2:
            raise NotImplementedError("Featurizer needs to be implemented for binary relations!")

        return feature_generators

class TableNgramPairFeaturizer(TableNgramFeaturizer):
    def _prepend_entity_label(self, generator, entity_label):
        for feat in generator:
            yield (feat[0], ('e%s_' % entity_label) + feat[1])

    def _match_contexts(self, candidates):
        # TODO: generalize this to arity=N
        # collect (entity) feature generators from parent
        e0_feature_generators = TableNgramFeaturizer._match_contexts(
            self, [candidate.span0 for candidate in candidates])
        e1_feature_generators = TableNgramFeaturizer._match_contexts(
            self, [candidate.span1 for candidate in candidates])

        feature_generators = []
        for i in range(len(e0_feature_generators)):
            feature_generators += [self._prepend_entity_label(e0_feature_generators[i],0)]
        for i in range(len(e1_feature_generators)):
            feature_generators += [self._prepend_entity_label(e1_feature_generators[i],1)]


        feature_generators.append(self._generate_context_feats( \
            lambda c : get_relation_table_feats(c), 'RTABLE_', candidates))

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
