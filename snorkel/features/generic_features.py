from functools import partial


def feats_from_matrix_generator(candidate, candidate_index, X, prefix):
    i = candidate_index[candidate.id]
    for j in xrange(X.shape[1]):
        yield '{0}_{1}'.format(prefix, j), X[i, j]


def get_feats_from_matrix(candidate_index, X, prefix='col'):
    return partial(feats_from_matrix_generator, candidate_index=candidate_index,
    	X=X, prefix=prefix)
