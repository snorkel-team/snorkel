import re
import sys
import numpy as np
import scipy.sparse as sparse


class ProgressBar(object):
    def __init__(self, N, length=40):
        self.N      = max(1, N)
        self.nf     = float(self.N)
        self.length = length


    def bar(self, i):
        """Assumes i ranges through [0, N-1]"""
        b = int(np.ceil(((i+1) / self.nf) * self.length))
        sys.stdout.write("\r[%s%s] %d%%" % ("="*b, " "*(self.length-b), int(100*((i+1) / self.nf))))
        sys.stdout.flush()

    def close(self):
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


def matrix_accuracy(L, G):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate
    and a set of gold candidates:
    Return the accuracy of each LF compared to the gold.
    Accuracy is defined as: (# correct non-zero labels) / (# non-zero labels)
    """
    N, M = L.shape
    pb = ProgressBar(len(G))
    # Create N x 1 vector to compare against
    gold_labels = np.ndarray((N,1))
    gold_labels.fill(-1)
    count = 0
    for c in G:
        pb.bar(count)
        count += 1
        index = L.get_row_index(c) # NOTE: Assumes G is a subset of L
        gold_labels[index] = [1]

    pb.close()
    gold_label_matrix = np.repeat(gold_labels, M, axis=1)
    correct_labels = np.clip(np.multiply(gold_label_matrix, L.toarray()), 0, 1)
    # Calculate accuracy of each LF
    non_zero_cols = (L.toarray() != 0).sum(axis=0) # count non-zero elements of each col
    accuracy = np.divide(np.sum(correct_labels, axis=0), non_zero_cols)
    return np.ravel(accuracy)


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


def split_html_attrs(attrs):
    """
    Given an iterable object of (attr, values) pairs, returns a list of separated
    "attr=value" strings
    """
    html_attrs = []
    for a in attrs:
        attr = a[0]
        values = [v.split(';') for v in a[1]] if isinstance(a[1],list) else [a[1].split(';')]
        for i in range(len(values)):
            while isinstance(values[i], list):
                values[i] = values[i][0]
        html_attrs += ["=".join([attr,val]) for val in values]
    return html_attrs


def tokens_to_ngrams(tokens, n_min=1, n_max=3, delim=' '):
    N = len(tokens)
    for root in range(N):
        for n in range(max(n_min - 1, 0), min(n_max, N - root)):
            yield delim.join(tokens[root:root+n+1])


def get_keys_by_candidate(annotation_matrix, candidate):
    (r,c,v) = sparse.find(annotation_matrix[annotation_matrix.get_row_index(candidate),:])
    return [annotation_matrix.get_key(idx) for idx in c]
