from itertools import chain
from typing import List, Tuple

# NB: don't include pyspark in requirements.txt to avoid
# overwriting existing system Spark install
import scipy.sparse as sparse
from pyspark import RDD

from snorkel.types import DataPoint

from .lf_applier import BaseLFApplier

RowData = List[Tuple[int, int, int]]


class SparkLFApplier(BaseLFApplier):
    def apply(self, data_points: RDD) -> List[RowData]:  # type: ignore
        def map_fn(data_tuple: Tuple[DataPoint, int]) -> RowData:
            x, i = data_tuple
            labels = []
            for j, lf in enumerate(self._lfs):
                y = lf(x)
                if y != 0:
                    labels.append((i, j, y))
            return labels

        labels = data_points.zipWithIndex().map(map_fn).collect()
        n, m = len(labels), len(self._lfs)
        row, col, data = zip(*chain.from_iterable(labels))
        return sparse.csr_matrix((data, (row, col)), shape=(n, m))
