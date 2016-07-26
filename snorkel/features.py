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
    def __init__(self, arity):
        self.arity = arity
    
    def _generate_context_feats(self, get_feats, prefix, candidates):
        """
        Given a function that given a candidate, generates features, _using a specific context_,
        and a unique prefix string for this context, return a generator over features (as strings).
        """
        for i,c in enumerate(candidates):
            for f in get_feats(c):
                yield i, prefix + f

    def apply(self, candidates):
        feature_generators = []

        # TODO: This is *TEMP* code! Also, this is for legacy candidates!!
        if self.arity == 1:
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_ddlib_feats(c, c.idxs), 'DDLIB_', candidates))
        if self.arity == 1 and candidates[0].root is not None:
            get_feats = compile_entity_feature_generator()
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_feats(c.root, c.idxs), 'TDLIB_', candidates))
        if self.arity == 2 and candidates[0].root is not None:
            get_feats = compile_relation_feature_generator()
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_feats(c.root, c.e1_idxs, c.e2_idxs), 'TDLIB_', candidates))

        # Assemble and return the sparse feature matrix
        f_index = defaultdict(list)
        for i,f in itertools.chain(*feature_generators):
            f_index[f].append(i)

        # Assemble and return sparse feature matrix
        # Also assemble reverse index of feature matrix index -> feature verbose name
        self.feature_index = {}
        F                  = sparse.lil_matrix((len(candidates), len(f_index.keys())))
        for j,f in enumerate(f_index.keys()):
            self.feature_index[j] = f
            for i in f_index[f]:
                F[i,j] = 1
        return F
