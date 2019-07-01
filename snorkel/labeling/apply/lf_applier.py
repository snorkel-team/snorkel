from itertools import chain
from typing import Any, List, Tuple

import scipy.sparse as sparse
from tqdm import tqdm

from snorkel.labeling.lf import LabelingFunction
from snorkel.labeling.preprocess import PreprocessorMode
from snorkel.types import DataPoint, DataPoints

RowData = List[Tuple[int, int, int]]


class BaseLFApplier:
    """Base class for LF applier objects.

    Base class for LF applier objects, which executes a set of LFs
    on a collection of data points. Subclasses should operate on
    a single data point collection format (e.g. `DataFrame`).
    Subclasses must implement the `apply` method.

    Parameters
    ----------
    lfs
        LFs that this applier executes on examples

    Raises
    ------
    NotImplementedError
        `apply` method must be implemented by subclasses
    """

    def __init__(self, lfs: List[LabelingFunction]) -> None:
        self._lfs = lfs

    def _matrix_from_row_data(self, labels: List[RowData]) -> sparse.csr_matrix:
        row, col, data = zip(*chain.from_iterable(labels))
        n, m = len(labels), len(self._lfs)
        return sparse.csr_matrix((data, (row, col)), shape=(n, m))

    def _set_lf_preprocessor_mode(self, mode: PreprocessorMode) -> None:
        for lf in self._lfs:
            lf.set_preprocessor_mode(mode)

    def apply(self, data_points: Any, *args: Any, **kwargs: Any) -> sparse.csr_matrix:
        """Label collection of data points with LFs.

        Parameters
        ----------
        data_points
            Collection of data points to be labeled by LFs. Subclasses
            implement functionality for a specific format (e.g. `DataFrame`).

        Returns
        -------
        sparse.csr_matrix
            Sparse matrix of labels emitted by LFs

        Raises
        ------
        NotImplementedError
            `apply` method must be implemented by subclasses
        """
        raise NotImplementedError


def apply_lfs_to_data_point(
    x: DataPoint, index: int, lfs: List[LabelingFunction]
) -> RowData:
    """Label a single data point with a set of LFs.

    Parameters
    ----------
    x
        Data point to label
    index
        Index of the data point
    lfs
        Set of LFs to label `x` with

    Returns
    -------
    RowData
        A list of (data point index, LF index, label) tuples
    """
    labels = []
    for j, lf in enumerate(lfs):
        y = lf(x)
        if y != 0:
            labels.append((index, j, y))
    return labels


class LFApplier(BaseLFApplier):
    """LF applier for a list of data points.

    Labels a list of data points (e.g. `SimpleNamespace`). Primarily
    useful for testing.
    """

    def apply(self, data_points: DataPoints) -> sparse.csr_matrix:  # type: ignore
        """Label list of data points with LFs.

        Parameters
        ----------
        data_points
            List of data points to be labeled by LFs

        Returns
        -------
        sparse.csr_matrix
            Sparse matrix of labels emitted by LFs
        """
        self._set_lf_preprocessor_mode(PreprocessorMode.NAMESPACE)
        labels = []
        for i, x in tqdm(enumerate(data_points)):
            labels.append(apply_lfs_to_data_point(x, i, self._lfs))
        return self._matrix_from_row_data(labels)
