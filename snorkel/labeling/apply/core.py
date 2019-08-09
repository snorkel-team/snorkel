from itertools import chain
from typing import List, Tuple, Union

import numpy as np
from tqdm import tqdm

from snorkel.labeling.lf import LabelingFunction
from snorkel.types import DataPoint, DataPoints
from snorkel.utils.data_operators import check_unique_names

RowData = List[Tuple[int, int, int]]


class BaseLFApplier:
    """Base class for LF applier objects.

    Base class for LF applier objects, which executes a set of LFs
    on a collection of data points. Subclasses should operate on
    a single data point collection format (e.g. ``DataFrame``).
    Subclasses must implement the ``apply`` method.

    Parameters
    ----------
    lfs
        LFs that this applier executes on examples

    Raises
    ------
    ValueError
        If names of LFs are not unique
    """

    def __init__(self, lfs: List[LabelingFunction]) -> None:
        self._lfs = lfs
        self._lf_names = [lf.name for lf in lfs]
        check_unique_names(self._lf_names)

    def _matrix_from_row_data(self, labels: List[RowData]) -> np.ndarray:
        L = np.zeros((len(labels), len(self._lfs)), dtype=int) - 1
        # NB: this check will short-circuit, so ok for large L
        if any(map(len, labels)):
            row, col, data = zip(*chain.from_iterable(labels))
            L[row, col] = data
        return L

    def __repr__(self) -> str:
        return f"{type(self).__name__}, LFs: {self._lf_names}"


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
        Set of LFs to label ``x`` with

    Returns
    -------
    RowData
        A list of (data point index, LF index, label) tuples
    """
    labels = []
    for j, lf in enumerate(lfs):
        y = lf(x)
        if y >= 0:
            labels.append((index, j, y))
    return labels


class LFApplier(BaseLFApplier):
    """LF applier for a list of data points (e.g. ``SimpleNamespace``) or a NumPy array.

    Parameters
    ----------
    lfs
        LFs that this applier executes on examples

    Example
    -------
    >>> from snorkel.labeling import labeling_function
    >>> @labeling_function()
    ... def is_big_num(x):
    ...     return 1 if x.num > 42 else 0
    >>> applier = LFApplier([is_big_num])
    >>> from types import SimpleNamespace
    >>> applier.apply([SimpleNamespace(num=10), SimpleNamespace(num=100)])
    array([[0], [1]])

    >>> @labeling_function()
    ... def is_big_num_np(x):
    ...     return 1 if x[0] > 42 else 0
    >>> applier = LFApplier([is_big_num_np])
    >>> applier.apply(np.array([[10], [100]]))
    array([[0], [1]])
    """

    def apply(
        self, data_points: Union[DataPoints, np.ndarray], progress_bar: bool = True
    ) -> np.ndarray:
        """Label list of data points or a NumPy array with LFs.

        Parameters
        ----------
        data_points
            List of data points or NumPy array to be labeled by LFs
        progress_bar
            Display a progress bar?

        Returns
        -------
        np.ndarray
            Matrix of labels emitted by LFs
        """
        labels = []
        for i, x in tqdm(enumerate(data_points), disable=(not progress_bar)):
            labels.append(apply_lfs_to_data_point(x, i, self._lfs))
        return self._matrix_from_row_data(labels)
