from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from functools import partial


def feats_from_matrix_generator(candidate, candidate_index, X, prefix):
    """Dump a precomputed feature matrix directly into Snorkel
    candidate: @Candidate to extract features for
    candidate_index: map from @Candidate to row index in @X
    X: matrix of features
    prefix: @string with feature key prefix
    """
    i = candidate_index[candidate.id]
    for j in range(X.shape[1]):
        yield '{0}_{1}'.format(prefix, j), X[i, j]


def get_feats_from_matrix(candidate_index, X, prefix='col'):
    """Get a feature matrix generator unary function"""
    return partial(feats_from_matrix_generator, candidate_index=candidate_index,
        X=X, prefix=prefix)
