from functools import partial
from typing import Union

import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client

from .core import BaseLFApplier, _FunctionCaller
from .pandas import apply_lfs_to_data_point, rows_to_triplets

Scheduler = Union[str, Client]


class DaskLFApplier(BaseLFApplier):
    """LF applier for a Dask DataFrame.

    Dask DataFrames consist of partitions, each being a Pandas DataFrame.
    This allows for efficient parallel computation over DataFrame rows.
    For more information, see https://docs.dask.org/en/stable/dataframe.html
    """

    def apply(
        self,
        df: dd.DataFrame,
        scheduler: Scheduler = "processes",
        fault_tolerant: bool = False,
    ) -> np.ndarray:
        """Label Dask DataFrame of data points with LFs.

        Parameters
        ----------
        df
            Dask DataFrame containing data points to be labeled by LFs
        scheduler
            A Dask scheduling configuration: either a string option or
            a ``Client``. For more information, see
            https://docs.dask.org/en/stable/scheduling.html#
        fault_tolerant
            Output ``-1`` if LF execution fails?

        Returns
        -------
        np.ndarray
            Matrix of labels emitted by LFs
        """
        f_caller = _FunctionCaller(fault_tolerant)
        apply_fn = partial(apply_lfs_to_data_point, lfs=self._lfs, f_caller=f_caller)
        map_fn = df.map_partitions(lambda p_df: p_df.apply(apply_fn, axis=1))
        labels = map_fn.compute(scheduler=scheduler)
        labels_with_index = rows_to_triplets(labels)
        return self._numpy_from_row_data(labels_with_index)


class PandasParallelLFApplier(DaskLFApplier):
    """Parallel LF applier for a Pandas DataFrame.

    Creates a Dask DataFrame from a Pandas DataFrame, then uses
    ``DaskLFApplier`` to label data in parallel. See ``DaskLFApplier``.
    """

    def apply(  # type: ignore
        self,
        df: pd.DataFrame,
        n_parallel: int = 2,
        scheduler: Scheduler = "processes",
        fault_tolerant: bool = False,
    ) -> np.ndarray:
        """Label Pandas DataFrame of data points with LFs in parallel using Dask.

        Parameters
        ----------
        df
            Pandas DataFrame containing data points to be labeled by LFs
        n_parallel
            Parallelism level for LF application. Corresponds to ``npartitions``
            in constructed Dask DataFrame. For ``scheduler="processes"``, number
            of processes launched. Recommended to be no more than the number
            of cores on the running machine.
        scheduler
            A Dask scheduling configuration: either a string option or
            a ``Client``. For more information, see
            https://docs.dask.org/en/stable/scheduling.html#
        fault_tolerant
            Output ``-1`` if LF execution fails?

        Returns
        -------
        np.ndarray
            Matrix of labels emitted by LFs
        """
        if n_parallel < 2:
            raise ValueError(
                "n_parallel should be >= 2. "
                "For single process Pandas, use PandasLFApplier."
            )
        df = dd.from_pandas(df, npartitions=n_parallel)
        return super().apply(df, scheduler=scheduler, fault_tolerant=fault_tolerant)
