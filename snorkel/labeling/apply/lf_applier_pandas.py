from functools import partial
from typing import List, Tuple

import scipy.sparse as sparse
from pandas import DataFrame
from tqdm import tqdm

from snorkel.labeling.lf import LabelingFunction
from snorkel.types import Example

from .lf_applier import BaseLFApplier

PandasRowData = List[Tuple[int, int]]


def apply_lfs_to_example(
    example: Example, lfs: List[LabelingFunction]
) -> PandasRowData:
    labels = []
    for j, lf in enumerate(lfs):
        y = lf(example)
        if y != 0:
            labels.append((j, y))
    return labels


class PandasLFApplier(BaseLFApplier):
    def apply(self, df: DataFrame) -> sparse.csr_matrix:  # type: ignore
        apply_fn = partial(apply_lfs_to_example, lfs=self._lfs)
        tqdm.pandas()
        labels = df.progress_apply(apply_fn, axis=1)
        labels_with_index = [
            [(index, j, y) for j, y in row_labels]
            for index, row_labels in enumerate(labels)
        ]
        return self._matrix_from_row_data(labels_with_index)
