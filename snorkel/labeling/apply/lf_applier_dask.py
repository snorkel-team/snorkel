from functools import partial
from typing import Union

import scipy.sparse as sparse
from dask import dataframe as DataFrame
from dask.distributed import Client

from snorkel.labeling.preprocess import PreprocessorMode

from .lf_applier import BaseLFApplier
from .lf_applier_pandas import apply_lfs_to_data_point, rows_to_triplets

Scheduler = Union[str, Client]


class DaskLFApplier(BaseLFApplier):
    """LF applier for a Dask DataFrame.

    Dask DataFrames consist of partitions, each being a Pandas DataFrame.
    This allows for efficient parallel computation over DataFrame rows.
    For more information, see https://docs.dask.org/en/stable/dataframe.html
    """

    def apply(
        self, df: DataFrame, scheduler: Scheduler = "processes"
    ) -> sparse.csr_matrix:  # type: ignore
        """Label Dask DataFrame of data points with LFs.

        Parameters
        ----------
        data_points
            Dask DataFrame containing data points to be labeled by LFs
        scheduler
            A Dask scheduling configuration: either a string option or
            a `Client`. For more information, see
            https://docs.dask.org/en/stable/scheduling.html#

        Returns
        -------
        sparse.csr_matrix
            Sparse matrix of labels emitted by LFs
        """
        self._set_lf_preprocessor_mode(PreprocessorMode.DASK)
        apply_fn = partial(apply_lfs_to_data_point, lfs=self._lfs)
        map_fn = df.map_partitions(lambda p_df: p_df.apply(apply_fn, axis=1))
        labels = map_fn.compute(scheduler=scheduler)
        labels_with_index = rows_to_triplets(labels)
        return self._matrix_from_row_data(labels_with_index)
