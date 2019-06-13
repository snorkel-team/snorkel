from typing import List, Tuple

# NB: don't include pyspark in requirements.txt to avoid
# overwriting existing system Spark install
from pyspark import RDD

from snorkel.types import Example

from .lf_applier import BaseLFApplier, RowData, apply_lfs_to_example


class SparkLFApplier(BaseLFApplier):
    def apply(self, examples: RDD) -> List[RowData]:  # type: ignore
        def _apply_instance_lfs_to_example_star(args: Tuple[Example, int]) -> RowData:
            return apply_lfs_to_example(*args, lfs=self._lfs)

        lf_map = examples.zipWithIndex().map(_apply_instance_lfs_to_example_star)
        labels = lf_map.collect()
        return self._matrix_from_row_data(labels)
