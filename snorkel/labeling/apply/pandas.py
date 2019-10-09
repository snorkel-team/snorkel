from functools import partial
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from snorkel.labeling.lf import LabelingFunction
from snorkel.types import DataPoint

from .core import ApplierMetadata, BaseLFApplier, RowData, _FunctionCaller

PandasRowData = List[Tuple[int, int]]


def apply_lfs_to_data_point(
    x: DataPoint, lfs: List[LabelingFunction], f_caller: _FunctionCaller
) -> PandasRowData:
    """Label a single data point with a set of LFs.

    Parameters
    ----------
    x
        Data point to label
    lfs
        Set of LFs to label ``x`` with
    f_caller
        A ``_FunctionCaller`` to record failed LF executions

    Returns
    -------
    RowData
        A list of (LF index, label) tuples
    """
    labels = []
    for j, lf in enumerate(lfs):
        y = f_caller(lf, x)
        if y >= 0:
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
    >>> applier = PandasLFApplier([is_big_num])
    >>> applier.apply(pd.DataFrame(dict(num=[10, 100], text=["hello", "hi"])))
    array([[0], [1]])
    """

    def apply(
        self,
        df: pd.DataFrame,
        progress_bar: bool = True,
        fault_tolerant: bool = False,
        return_meta: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ApplierMetadata]]:
        """Label Pandas DataFrame of data points with LFs.

        Parameters
        ----------
        df
            Pandas DataFrame containing data points to be labeled by LFs
        progress_bar
            Display a progress bar?
        fault_tolerant
            Output ``-1`` if LF execution fails?
        return_meta
            Return metadata from apply call?

        Returns
        -------
        np.ndarray
            Matrix of labels emitted by LFs
        ApplierMetadata
            Metadata, such as fault counts, for the apply call
        """
        f_caller = _FunctionCaller(fault_tolerant)
        apply_fn = partial(apply_lfs_to_data_point, lfs=self._lfs, f_caller=f_caller)
        call_fn = df.apply
        if progress_bar:
            tqdm.pandas()
            call_fn = df.progress_apply
        labels = call_fn(apply_fn, axis=1)
        labels_with_index = rows_to_triplets(labels)
        L = self._numpy_from_row_data(labels_with_index)
        if return_meta:
            return L, ApplierMetadata(f_caller.fault_counts)
        return L
