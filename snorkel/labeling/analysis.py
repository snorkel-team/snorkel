from typing import List, Optional, Union

import numpy as np
import scipy.sparse as sparse
from pandas import DataFrame, Series

from snorkel.analysis.error_analysis import confusion_matrix
from snorkel.analysis.utils import arraylike_to_numpy
from snorkel.types import ArrayLike

Matrix = Union[np.ndarray, sparse.csr_matrix]


def _covered_data_points(L: Matrix) -> np.ndarray:
    """Get indicator vector z where z_i = 1 if x_i is labeled by at least one LF."""
    return np.ravel(np.where(L.sum(axis=1) != 0, 1, 0))


def _overlapped_data_points(L: Matrix) -> np.ndarray:
    """Get indicator vector z where z_i = 1 if x_i is labeled by more than one LF."""
    return np.where(np.ravel((L != 0).sum(axis=1)) > 1, 1, 0)


def _conflicted_data_points(L: sparse.spmatrix) -> np.ndarray:
    """Get indicator vector z where z_i = 1 if x_i is labeled differently by two LFs."""
    m = sparse.diags(np.ravel(L.max(axis=1).todense()))
    return np.ravel(np.max(m @ (L != 0) != L, axis=1).astype(int).todense())


def label_coverage(L: Matrix) -> float:
    """Compute the fraction of data points with at least one label.

    Parameters
    ----------
    L
        Matrix where L_{i,j} is the label given by the jth LF to the ith candidate

    Returns
    -------
    float
        Fraction of data points with labels
    """
    return _covered_data_points(L).sum() / L.shape[0]


def label_overlap(L: Matrix) -> float:
    """Compute the fraction of data points with at least two (non-abstain) labels.

    Parameters
    ----------
    L
        Matrix where L_{i,j} is the label given by the jth LF to the ith candidate

    Returns
    -------
    float
        Fraction of data points with overlapping labels
    """
    return _overlapped_data_points(L).sum() / L.shape[0]


def label_conflict(L: sparse.spmatrix) -> float:
    """Compute the fraction of data points with conflicting (non-abstain) labels.

    Parameters
    ----------
    L
        Matrix where L_{i,j} is the label given by the jth LF to the ith candidate

    Returns
    -------
    float
        Fraction of data points with conflicting labels
    """
    return _conflicted_data_points(L).sum() / L.shape[0]


def lf_polarities(L: Matrix) -> List[List[int]]:
    """Infer the polarities of each LF based on evidence in a label matrix.

    Parameters
    ----------
    L
        Matrix where L_{i,j} is the label given by the jth LF to the ith candidate

    Returns
    -------
    List[List[int]]
        Unique output labels for each LF
    """
    return [sorted(list(set(L[:, i].data))) for i in range(L.shape[1])]


def lf_coverages(L: Matrix) -> np.ravel:
    """Compute frac. of examples each LF labels.

    Parameters
    ----------
    L
        Matrix where L_{i,j} is the label given by the jth LF to the ith candidate

    Returns
    -------
    numpy.ndarray
        Fraction of labeled examples for each LF
    """
    return np.ravel((L != 0).sum(axis=0)) / L.shape[0]


def lf_overlaps(L: Matrix, normalize_by_coverage: bool = False) -> np.ndarray:
    """Compute frac. of examples each LF labels that are labeled by another LF.

    An overlapping example is one that at least one other LF returns a
    (non-abstain) label for.

    Note that the maximum possible overlap fraction for an LF is the LF's
    coverage, unless `normalize_by_coverage=True`, in which case it is 1.

    Parameters
    ----------
    L
        Matrix where L_{i,j} is the label given by the jth LF to the ith candidate
    normalize_by_coverage
        Normalize by coverage of the LF, so that it returns the percent of LF labels
        that have overlaps.

    Returns
    -------
    numpy.ndarray
        Fraction of overlapping examples for each LF
    """
    overlaps = (L != 0).T @ _overlapped_data_points(L) / L.shape[0]
    if normalize_by_coverage:
        overlaps /= lf_coverages(L)
    return np.nan_to_num(overlaps)


def lf_conflicts(L: sparse.spmatrix, normalize_by_overlaps: bool = False) -> np.ndarray:
    """Compute frac. of examples each LF labels and labeled differently by another LF.

    A conflicting example is one that at least one other LF returns a
    different (non-abstain) label for.

    Note that the maximum possible conflict fraction for an LF is the LF's
    overlaps fraction, unless `normalize_by_overlaps=True`, in which case it is 1.

    Parameters
    ----------
    L
        Matrix where L_{i,j} is the label given by the jth LF to the ith candidate
    normalize_by_overlaps
        Normalize by overlaps of the LF, so that it returns the percent of LF
        overlaps that have conflicts.

    Returns
    -------
    numpy.ndarray
        Fraction of conflicting examples for each LF
    """
    conflicts = (L != 0).T @ _conflicted_data_points(L) / L.shape[0]
    if normalize_by_overlaps:
        conflicts /= lf_overlaps(L)
    return np.nan_to_num(conflicts)


def lf_empirical_accuracies(L: Matrix, Y: ArrayLike) -> np.ndarray:
    """Compute empirical accuracy against a set of labels Y for each LF.

    Usually, Y represents development set labels.

    Parameters
    ----------
    L
        Matrix where L_{i,j} is the label given by the jth LF to the ith candidate
    Y
        [n] or [n, 1] np.ndarray of gold labels

    Returns
    -------
    numpy.ndarray
        Empirical accuracies for each LF
    """
    # Assume labeled set is small, work with dense matrices
    Y = arraylike_to_numpy(Y)
    L = L.toarray()
    X = np.where(L == 0, 0, np.where(L == np.vstack([Y] * L.shape[1]).T, 1, -1))
    return np.nan_to_num(0.5 * (X.sum(axis=0) / (L != 0).sum(axis=0) + 1))


def lf_summary(
    L: Matrix,
    Y: Optional[ArrayLike] = None,
    lf_names: Optional[Union[List[str], List[int]]] = None,
    est_accs: Optional[np.ndarray] = None,
) -> DataFrame:
    """Create a pandas DataFrame with the various per-LF statistics.

    Parameters
    ----------
    L
        Matrix where L_{i,j} is the label given by the jth LF to the ith candidate
    Y
        [n] or [n, 1] np.ndarray of gold labels. If provided, the empirical accuracy
        for each LF will be calculated.
    lf_names
        Name of each LF. If None, indices are used.
    est_accs
        Learned accuracies for each LF

    Returns
    -------
    pandas.DataFrame
        Summary statistics for each LF
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
    """Calculate coverage, overlap, conflicts, and accuracy for a single LF.

    Parameters
    ----------
    Y_p
        Array of predicted labels
    Y
        Array of true labels (if known)

    Returns
    -------
    pandas.DataFrame
        Summary statistics for LF
    """
    L = sparse.csr_matrix(arraylike_to_numpy(Y_p).reshape(-1, 1))
    summary = lf_summary(L, Y)
    return summary[["Polarity", "Coverage", "Correct", "Incorrect", "Emp. Acc."]]
