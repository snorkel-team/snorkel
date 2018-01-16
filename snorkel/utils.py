from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import re
import sys
import numpy as np
import scipy.sparse as sparse
import subprocess


class ProgressBar(object):
    def __init__(self, N, length=40):
        self.N      = max(1, N)
        self.nf     = float(self.N)
        self.length = length
        self.update_interval = self.nf/100
        self.current_tick = 0
        self.bar(0)

    def bar(self, i):
        """Assumes i ranges through [0, N-1]"""
        new_tick = i/self.update_interval
        if int(new_tick) != int(self.current_tick):
            b = int(np.ceil((i / self.nf) * self.length))
            sys.stdout.write("\r[%s%s] %d%%" % ("="*b, " "*(self.length-b), int(100*(i / self.nf))))
            sys.stdout.flush()
        self.current_tick = new_tick

    def close(self):
        b = self.length
        sys.stdout.write("\r[%s%s] %d%%\n" % ("="*b, " "*(self.length-b), 100))
        sys.stdout.flush()


def get_ORM_instance(ORM_class, session, instance):
    """
    Given an ORM class and *either an instance of this class, or the name attribute of an instance
    of this class*, return the instance
    """
    if isinstance(instance, str) or isinstance(instance, unicode):
        return session.query(ORM_class).filter(ORM_class.name == instance).one_or_none()
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

def sparse_nonzero(X):
    """Sparse matrix with value 1 for i,jth entry !=0"""
    X_nonzero = X.copy()
    if not sparse.issparse(X):
        X_nonzero[X_nonzero != 0] = 1
        return X_nonzero
    if sparse.isspmatrix_csr(X) or sparse.isspmatrix_csc(X):
        X_nonzero.data[X_nonzero.data != 0] = 1
    elif sparse.isspmatrix_lil(X):
        X_nonzero.data = [np.ones(len(L)) for L in X_nonzero.data]
    else:
        raise ValueError("Only supports CSR/CSC and LIL matrices")
    return X_nonzero

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
    return np.ravel(sparse_nonzero(L).sum(axis=0) / float(L.shape[0]))


def matrix_overlaps(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _overlaps with other LFs on_.**
    """
    L_nonzero = sparse_nonzero(L)
    return np.ravel(np.where(L_nonzero.sum(axis=1) > 1, 1, 0).T * L_nonzero / float(L.shape[0]))

def matrix_conflicts(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _conflicts with other LFs on_.**
    """
    B = L.copy()
    if not sparse.issparse(B):
        for row in range(B.shape[0]):
            if np.unique(np.array(B[row][np.nonzero(B[row])])).size == 1:
                B[row] = 0
        return matrix_coverage(sparse_nonzero(B))
    if not (sparse.isspmatrix_csc(B) or sparse.isspmatrix_lil(B) or sparse.isspmatrix_csr(B)):
        raise ValueError("Only supports CSR/CSC and LIL matrices")
    if sparse.isspmatrix_csc(B) or sparse.isspmatrix_lil(B):
        B = B.tocsr()
    for row in range(B.shape[0]):
        if np.unique(B.getrow(row).data).size == 1:
            B.data[B.indptr[row]:B.indptr[row+1]] = 0
    return matrix_coverage(sparse_nonzero(B))



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
  return [d[w] if w in d else w for w in words]


def split_html_attrs(attrs):
    """
    Given an iterable object of (attr, values) pairs, returns a list of separated
    "attr=value" strings
    """
    html_attrs = []
    for a in attrs:
        attr = a[0]
        values = [v.split(';') for v in a[1]] if isinstance(a[1], list) else [a[1].split(';')]
        for i in range(len(values)):
            while isinstance(values[i], list):
                values[i] = values[i][0]
        html_attrs += ["=".join([attr, val]) for val in values]
    return html_attrs


def tokens_to_ngrams(tokens, n_min=1, n_max=3, delim=' ', lower=False):
    f = (lambda x: x.lower()) if lower else (lambda x: x)
    N = len(tokens)
    for root in range(N):
        for n in range(max(n_min - 1, 0), min(n_max, N - root)):
            yield f(delim.join(tokens[root:root + n + 1]))


def get_keys_by_candidate(candidate, annotation_matrix):
    (r, c, v) = sparse.find(annotation_matrix[annotation_matrix.get_row_index(candidate), :])
    return [annotation_matrix.get_key(idx).name for idx in c]


def remove_files(filename):
    try:
        subprocess.check_output('rm -f %s' % filename, shell=True)
    except OSError as e:
        print(e)
        pass
