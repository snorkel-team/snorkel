from functools import partial
from typing import List, Tuple

import numpy as np
from pandas import DataFrame
from tqdm import tqdm

from snorkel.labeling.lf import LabelingFunction
from snorkel.types import DataPoint

from .core import BaseLFApplier, RowData

PandasRowData = List[Tuple[int, int]]


def apply_lfs_to_data_point(x: DataPoint, lfs: List[LabelingFunction]) -> PandasRowData:
    """Label a single data point with a set of LFs.

    Parameters
    ----------
    x
        Data point to label
    lfs
        Set of LFs to label ``x`` with

    Returns
    -------
    RowData
        A list of (LF index, label) tuples
    """
    labels = []
    for j, lf in enumerate(lfs):
        y = lf(x)
        if y is not None:
            labels.append((j, y))
    return labels


def rows_to_triplets(labels: List[PandasRowData]) -> List[RowData]:
    """Convert list of list sparse matrix representation to list of triplets."""
    return [
        [(index, j, y) for j, y in row_labels]
        for index, row_labels in enumerate(labels)
    ]


class PandasLFApplier(BaseLFApplier):
    """LF applier for a Pandas DataFrame.

    Data points are stored as ``Series`` in a DataFrame. The LFs
    are executed via a ``pandas.DataFrame.apply`` call, which
    is single-process and can be slow for large DataFrames.
    For large datasets, consider ``DaskLFApplier`` or ``SparkLFApplier``.
    """

    def apply(self, df: DataFrame) -> np.ndarray:
        """Label Pandas DataFrame of data points with LFs.

        Parameters
        ----------
        df
            Pandas DataFrame containing data points to be labeled by LFs

        Returns
        -------
        np.ndarray
            Matrix of labels emitted by LFs
        """
        apply_fn = partial(apply_lfs_to_data_point, lfs=self._lfs)
        tqdm.pandas()
        labels = df.progress_apply(apply_fn, axis=1)
        labels_with_index = rows_to_triplets(labels)
        return self._matrix_from_row_data(labels_with_index)
