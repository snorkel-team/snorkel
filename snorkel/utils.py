import re
import sys
import numpy as np
import scipy.sparse as sparse


class ProgressBar(object):
    def __init__(self, N, length=40):
        self.N      = N
        self.nf     = float(N)
        self.length = length

    def bar(self, i):
        """Assumes i ranges through [0, N-1]"""
        b = int(np.ceil(((i+1) / self.nf) * self.length))
        sys.stdout.write("\r[%s%s] %d%%" % ("="*b, " "*(self.length-b), int(100*((i+1) / self.nf))))
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n\n")
        sys.stdout.flush()


def get_ORM_instance(ORM_class, session, instance):
    """
    Given an ORM class and *either an instance of this class, or the name attribute of an instance
    of this class*, return the instance
    """
    if isinstance(instance, str):
        return session.query(ORM_class).filter(ORM_class.name == instance).one()
    else:
        return instance


def camel_to_under(name):
    """
    Converts camel-case string to lowercase string separated by underscores.

    Written by epost
    (http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case).

    :param name: String to be converted
    :return: new String with camel-case converted to lowercase, underscored
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def sparse_abs(X):
    """Element-wise absolute value of sparse matrix- avoids casting to dense matrix!"""
    X_abs = X.copy()
    if not sparse.issparse(X):
        return abs(X_abs)
    if sparse.isspmatrix_csr(X) or sparse.isspmatrix_csc(X):
        X_abs.data = np.abs(X_abs.data)
    elif sparse.isspmatrix_lil(X):
        X_abs.data = np.array([np.abs(L) for L in X_abs.data])
    else:
        raise ValueError("Only supports CSR/CSC and LIL matrices")
    return X_abs


def matrix_label_mask(L):
    """Return a copy of L with all non-zero entries replaced with 1"""
    nz = L.nonzero()
    return sparse.csr_matrix((np.ones(len(nz[0])), nz), shape=L.shape)


def matrix_coverage(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF labels.**
    """
    # NOTE: I have an old version of scipy, need to upgrade and replace with .getnnz(axis)!
    L_mask = matrix_label_mask(L)
    return np.ravel(L_mask.sum(axis=0) / float(L.shape[0]))


def matrix_overlaps(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _overlaps with other LFs on_.**
    """
    L_mask = matrix_label_mask(L)
    return np.ravel(np.where(L_mask.sum(axis=1) > 1, 1, 0).T * L_mask / float(L.shape[0]))


def matrix_conflict_rows(L):
    """
    Given an N x M matrix, return an N-dim array with entries 1 if corresponding row
    did not have consistent (i.e. all the same) non-zero entries.
    """
    conflicted  = np.zeros(L.shape[0])
    current_val = np.zeros(L.shape[0])
    for i, j in zip(*L.nonzero()):
        if conflicted[i] == 0:
            if current_val[i] == 0:
                current_val[i] = L[i,j]
            elif current_val[i] != L[i,j]:
                conflicted[i] = 1
    return conflicted


def matrix_conflicts(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _conflicts with other LFs on_.**
    """
    L_mask       = matrix_label_mask(L)
    L_conflicted = matrix_conflict_rows(L)
    return np.ravel(L_conflicted * L_mask / float(L.shape[0]))


def matrix_accuracy(L, labels):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate
    and an N x 1 vector where v_{i} is the gold label given to the ith candidate:
    Return the **fraction of candidates that each LF covered and agreed with the gold labels**
    """
    correct = np.zeros(L.shape[1])
    labeled = np.zeros(L.shape[1])
    for i, j in zip(*L.nonzero()):
        labeled[j] += 1
        if L[i,j] == labels[i]:
            correct[j] += 1
    return correct / labeled


def matrix_tp(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] == 1).todense()) * (labels == 1)) for j in xrange(L.shape[1])
    ])

def matrix_fp(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] == 1).todense()) * (labels != 1)) for j in xrange(L.shape[1])
    ])

def matrix_tn(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] == -1).todense()) * (labels == -1)) for j in xrange(L.shape[1])
    ])

def matrix_fn(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] == -1).todense()) * (labels != -1)) for j in xrange(L.shape[1])
    ])

def get_as_dict(x):
    """Return an object as a dictionary of its attributes"""
    if isinstance(x, dict):
        return x
    else:
        try:
            return x._asdict()
        except AttributeError:
            return x.__dict__


def sort_X_on_Y(X, Y):
    return [x for (y,x) in sorted(zip(Y,X), key=lambda t : t[0])]


def corenlp_cleaner(words):
  d = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{',
       '-RSB-': ']', '-LSB-': '['}
  return map(lambda w: d[w] if w in d else w, words)


def tokens_to_ngrams(tokens, n_max=3, delim=' '):
    N = len(tokens)
    for root in range(N):
        for n in range(min(n_max, N - root)):
            yield delim.join(tokens[root:root+n+1])
