from itertools import product
from typing import List, Optional, Union

import numpy as np
import scipy.sparse as sparse
from pandas import DataFrame, Series

from snorkel.analysis.error_analysis import confusion_matrix
from snorkel.analysis.utils import arraylike_to_numpy
from snorkel.types import ArrayLike

Matrix = Union[np.ndarray, sparse.csr_matrix]


############################################################
# Label Matrix Diagnostics
############################################################
def _covered_data_points(L: Matrix) -> np.ndarray:
    """Returns an indicator vector where ith element = 1 if x_i is labeled by at
    least one LF."""
    return np.ravel(np.where(L.sum(axis=1) != 0, 1, 0))


def _overlapped_data_points(L: Matrix) -> np.ndarray:
    """Returns an indicator vector where ith element = 1 if x_i is labeled by
    more than one LF."""
    return np.where(np.ravel((L != 0).sum(axis=1)) > 1, 1, 0)


def _conflicted_data_points(L: sparse.spmatrix) -> np.ndarray:
    """Returns an indicator vector where ith element = 1 if x_i is labeled by
    at least two LFs that give it disagreeing labels."""
    m = sparse.diags(np.ravel(L.max(axis=1).todense()))
    return np.ravel(np.max(m @ (L != 0) != L, axis=1).astype(int).todense())


def label_coverage(L: Matrix) -> float:
    """Returns the **fraction of data points with > 0 (non-zero) labels**
    Args:
        L: an n x m scipy.sparse matrix where L_{i,j} is the label given by the
            jth LF to the ith item
    """
    return _covered_data_points(L).sum() / L.shape[0]


def label_overlap(L: Matrix) -> float:
    """Returns the **fraction of data points with > 1 (non-zero) labels**
    Args:
        L: an n x m scipy.sparse matrix where L_{i,j} is the label given by the
            jth LF to the ith item
    """
    return _overlapped_data_points(L).sum() / L.shape[0]


def label_conflict(L: sparse.spmatrix) -> float:
    """Returns the **fraction of data points with conflicting (disagreeing)
    labels.**
    Args:
        L: an n x m scipy.sparse matrix where L_{i,j} is the label given by the
            jth LF to the ith item
    """
    return _conflicted_data_points(L).sum() / L.shape[0]


def lf_polarities(L: Matrix) -> List[List[int]]:
    """Return the polarities of each LF based on evidence in a label matrix.

    Args:
        L: an n x m scipy.sparse matrix where L_{i,j} is the label given by the
            jth LF to the ith candidate
    """
    return [sorted(list(set(L[:, i].data))) for i in range(L.shape[1])]


def lf_coverages(L: Matrix) -> np.ravel:
    """Return the **fraction of data points that each LF labels.**
    Args:
        L: an n x m scipy.sparse matrix where L_{i,j} is the label given by the
            jth LF to the ith candidate
    """
    return np.ravel((L != 0).sum(axis=0)) / L.shape[0]


def lf_overlaps(L: Matrix, normalize_by_coverage: bool = False) -> np.ndarray:
    """Return the **fraction of items each LF labels that are also labeled by at
     least one other LF.**

    Note that the maximum possible overlap fraction for an LF is the LF's
    coverage, unless `normalize_by_coverage=True`, in which case it is 1.

    Args:
        L: an n x m scipy.sparse matrix where L_{i,j} is the label given by the
            jth LF to the ith candidate
        normalize_by_coverage: Normalize by coverage of the LF, so that it
            returns the percent of LF labels that have overlaps.
    """
    overlaps = (L != 0).T @ _overlapped_data_points(L) / L.shape[0]
    if normalize_by_coverage:
        overlaps /= lf_coverages(L)
    return np.nan_to_num(overlaps)


def lf_conflicts(L: sparse.spmatrix, normalize_by_overlaps: bool = False) -> np.ndarray:
    """Return the **fraction of items each LF labels that are also given a
    different (non-abstain) label by at least one other LF.**

    Note that the maximum possible conflict fraction for an LF is the LF's
        overlaps fraction, unless `normalize_by_overlaps=True`, in which case it
        is 1.

    Args:
        L: an n x m scipy.sparse matrix where L_{i,j} is the label given by the
            jth LF to the ith candidate
        normalize_by_overlaps: Normalize by overlaps of the LF, so that it
            returns the percent of LF overlaps that have conflicts.
    """
    conflicts = (L != 0).T @ _conflicted_data_points(L) / L.shape[0]
    if normalize_by_overlaps:
        conflicts /= lf_overlaps(L)
    return np.nan_to_num(conflicts)


def lf_empirical_accuracies(L: Matrix, Y: ArrayLike) -> np.ndarray:
    """Return the **empirical accuracy** against a set of labels Y (e.g. dev
    set) for each LF.
    Args:
        L: an n x m scipy.sparse matrix where L_{i,j} is the label given by the
            jth LF to the ith candidate
        Y: an [n] or [n, 1] np.ndarray of gold labels
    """
    # Assume labeled set is small, work with dense matrices
    Y = arraylike_to_numpy(Y)
    L = L.toarray()
    X = np.where(L == 0, 0, np.where(L == np.vstack([Y] * L.shape[1]).T, 1, -1))
    return np.nan_to_num(0.5 * (X.sum(axis=0) / (L != 0).sum(axis=0) + 1))


def lf_empirical_probs(L: Matrix, Y: ArrayLike, k: Optional[int] = None) -> np.ndarray:
    """Returns the conditional probability tables, P(L | Y), for each of the labeling
    functions, computed empirically using the provided true labels Y.

    Parameters
    ----------
    L
        The n x m matrix of LF labels, where n is # of datapoints and m is # of LFs
    Y
        The n-dim array of true labels in {1,...,k}
    k
        The cardinality i.e. number of classes; if not provided, defaults to max value
        in Y

    Returns
    -------
    np.ndarray
        An m x (k+1) x k np.ndarray representing the m (k+1) x k conditional probability
        tables P_i, where P_i[l,y] represents P(LF_i = l | Y = y) empirically calculated
    """
    n, m = L.shape

    # Assume labeled set is small, work with dense matrices
    Y = arraylike_to_numpy(Y)
    L = L.toarray()

    # Infer cardinality if not provided
    if k is None:
        k = Y.max()

    # Compute empirical conditional probabilities
    # Note: Can do this more efficiently...
    P = np.zeros((m, k + 1, k))
    for y in range(1, k + 1):
        is_y = np.where(Y == y, 1, 0)
        for j, l in product(range(m), range(k + 1)):
            P[j, l, y - 1] = np.where(L[:, j] == l, 1, 0) @ is_y / is_y.sum()
    return P


def lf_summary(
    L: Matrix,
    Y: Optional[ArrayLike] = None,
    lf_names: Optional[Union[List[str], List[int]]] = None,
    est_accs: Optional[np.ndarray] = None,
) -> DataFrame:
    """Returns a pandas DataFrame with the various per-LF statistics.

    Args:
        L: an n x m scipy.sparse matrix where L_{i,j} is the label given by the
            jth LF to the ith candidate
        Y: an [n] or [n, 1] np.ndarray of gold labels.
            If provided, the empirical accuracy for each LF will be calculated
    """
    n, m = L.shape
    if lf_names is not None:
        col_names = ["j"]
        d = {"j": list(range(m))}
    else:
        lf_names = list(range(m))
        col_names = []
        d = {}

    # Default LF stats
    col_names.extend(["Polarity", "Coverage", "Overlaps", "Conflicts"])
    d["Polarity"] = Series(data=lf_polarities(L), index=lf_names)
    d["Coverage"] = Series(data=lf_coverages(L), index=lf_names)
    d["Overlaps"] = Series(data=lf_overlaps(L), index=lf_names)
    d["Conflicts"] = Series(data=lf_conflicts(L), index=lf_names)

    if Y is not None:
        col_names.extend(["Correct", "Incorrect", "Emp. Acc."])
        confusions = [
            confusion_matrix(Y, L[:, i], pretty_print=False) for i in range(m)
        ]
        corrects = [np.diagonal(conf).sum() for conf in confusions]
        incorrects = [
            conf.sum() - correct for conf, correct in zip(confusions, corrects)
        ]
        accs = lf_empirical_accuracies(L, Y)
        d["Correct"] = Series(data=corrects, index=lf_names)
        d["Incorrect"] = Series(data=incorrects, index=lf_names)
        d["Emp. Acc."] = Series(data=accs, index=lf_names)

    if est_accs is not None:
        col_names.append("Learned Acc.")
        d["Learned Acc."] = Series(est_accs, index=lf_names)

    return DataFrame(data=d, index=lf_names)[col_names]


def single_lf_summary(Y_p: ArrayLike, Y: Optional[ArrayLike] = None) -> DataFrame:
    """Calculates coverage, overlap, conflicts, and accuracy for a single LF

    Args:
        Y_p: a np.array or torch.Tensor of predicted labels
        Y: a np.array or torch.Tensor of true labels (if known)
    """
    L = sparse.csr_matrix(arraylike_to_numpy(Y_p).reshape(-1, 1))
    summary = lf_summary(L, Y)
    return summary[["Polarity", "Coverage", "Correct", "Incorrect", "Emp. Acc."]]
