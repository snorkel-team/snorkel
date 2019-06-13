from itertools import chain
from typing import Any, Iterable, List, Tuple

import scipy.sparse as sparse
from tqdm import tqdm

from snorkel.labeling.lf import LabelingFunction
from snorkel.types import Example

RowData = List[Tuple[int, int, int]]


class BaseLFApplier:
    def __init__(self, lfs: List[LabelingFunction]) -> None:
        if not all(isinstance(f, LabelingFunction) for f in lfs):
            raise ValueError("lfs must be a list of LabelingFunctions")
        self._lfs = lfs

    def _matrix_from_row_data(self, labels: List[RowData]) -> sparse.csr_matrix:
        row, col, data = zip(*chain.from_iterable(labels))
        n, m = len(labels), len(self._lfs)
        return sparse.csr_matrix((data, (row, col)), shape=(n, m))

    def apply(self, examples: Any, *args: Any, **kwargs: Any) -> sparse.csr_matrix:
        raise NotImplementedError


def apply_lfs_to_example(
    example: Example, index: int, lfs: List[LabelingFunction]
) -> RowData:
    labels = []
    for j, lf in enumerate(lfs):
        y = lf(example)
        if y != 0:
            labels.append((index, j, y))
    return labels


class LFApplier(BaseLFApplier):
    def apply(self, examples: Iterable[Example]) -> sparse.csr_matrix:  # type: ignore
        labels = []
        for i, example in tqdm(enumerate(examples)):
            labels.append(apply_lfs_to_example(example, i, self._lfs))
        return self._matrix_from_row_data(labels)
