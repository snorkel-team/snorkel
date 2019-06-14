from typing import Tuple

# NB: don't include pyspark in requirements.txt to avoid
# overwriting existing system Spark install
import scipy.sparse as sparse
from pyspark import RDD

from snorkel.types import DataPoint

from .lf_applier import BaseLFApplier, RowData, apply_lfs_to_data_point


class SparkLFApplier(BaseLFApplier):
    def apply(self, data_points: RDD) -> sparse.csr_matrix:  # type: ignore
        def map_fn(args: Tuple[DataPoint, int]) -> RowData:
            return apply_lfs_to_data_point(*args, lfs=self._lfs)

        labels = data_points.zipWithIndex().map(map_fn).collect()
        return self._matrix_from_row_data(labels)
