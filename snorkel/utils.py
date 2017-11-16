import re
import sys
import numpy as np
import scipy.sparse as sparse
from time import time


class PrintTimer:
    """Prints msg at start, total time taken at end."""
    def __init__(self, msg, prefix="###"):
        self.msg = msg
        self.prefix = prefix + " " if len(prefix) > 0 else prefix

    def __enter__(self):
        self.t0 = time()
        print("{0}{1}".format(self.prefix, self.msg))

    def __exit__(self, type, value, traceback):
        print ("{0}Done in {1:.1f}s.\n".format(self.prefix, time() - self.t0))


class ProgressBar(object):
    def __init__(self, N, length=40):
        # Protect against division by zero (N = 0 results in full bar being printed)
        self.N      = max(1, N)
        self.nf     = float(self.N)
        self.length = length
        # Precalculate the i values that should trigger a write operation
        self.ticks = set([round(i/100.0 * N) for i in range(101)])
        self.ticks.add(N-1)
        self.bar(0)

    def bar(self, i):
        """Assumes i ranges through [0, N-1]"""
        if i in self.ticks:
            b = int(np.ceil(((i+1) / self.nf) * self.length))
            sys.stdout.write(
                "\r[{0}{1}] {2}%".format(
                    "="*b, " "*(self.length-b), int(100*((i+1) / self.nf))))
            sys.stdout.flush()

    def close(self):
        # Move the bar to 100% before closing
        self.bar(self.N-1)
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


def matrix_coverage(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF labels.**
    """
    return np.ravel(sparse_abs(L).sum(axis=0) / float(L.shape[0]))


def matrix_overlaps(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _overlaps with other LFs on_.**
    """
    L_abs = sparse_abs(L)
    return np.ravel(np.where(L_abs.sum(axis=1) > 1, 1, 0).T * L_abs / float(L.shape[0]))


def matrix_conflicts(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _conflicts with other LFs on_.**
    """
    L_abs = sparse_abs(L)
    return np.ravel(np.where(L_abs.sum(axis=1) != sparse_abs(L.sum(axis=1)), 1, 0).T * L_abs / float(L.shape[0]))

def matrix_tp(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] == 1).todense()) * (labels == 1)) for j in range(L.shape[1])
    ])

def matrix_fp(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] == 1).todense()) * (labels == -1)) for j in range(L.shape[1])
    ])

def matrix_tn(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] == -1).todense()) * (labels == -1)) for j in range(L.shape[1])
    ])

def matrix_fn(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] == -1).todense()) * (labels == 1)) for j in range(L.shape[1])
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
