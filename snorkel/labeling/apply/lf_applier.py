from itertools import chain
from typing import Any, List, Tuple

import scipy.sparse as sparse
from tqdm import tqdm

from snorkel.labeling.lf import LabelingFunction
from snorkel.labeling.preprocess import PreprocessorMode
from snorkel.types import DataPoint, DataPoints

RowData = List[Tuple[int, int, int]]


class BaseLFApplier:
    def __init__(self, lfs: List[LabelingFunction]) -> None:
        self._lfs = lfs

    def _matrix_from_row_data(self, labels: List[RowData]) -> sparse.csr_matrix:
        row, col, data = zip(*chain.from_iterable(labels))
        n, m = len(labels), len(self._lfs)
        return sparse.csr_matrix((data, (row, col)), shape=(n, m))

    def _set_lf_preprocessor_mode(self, mode: PreprocessorMode) -> None:
        for lf in self._lfs:
            lf.set_preprocessor_mode(mode)

    def apply(self, data_points: Any, *args: Any, **kwargs: Any) -> sparse.csr_matrix:
        raise NotImplementedError


def apply_lfs_to_data_point(
    x: DataPoint, index: int, lfs: List[LabelingFunction]
) -> RowData:
    labels = []
    for j, lf in enumerate(lfs):
        y = lf(x)
        if y != 0:
            labels.append((index, j, y))
    return labels


class LFApplier(BaseLFApplier):
    def apply(self, data_points: DataPoints) -> sparse.csr_matrix:  # type: ignore
        self._set_lf_preprocessor_mode(PreprocessorMode.NAMESPACE)
        labels = []
        for i, x in tqdm(enumerate(data_points)):
            labels.append(apply_lfs_to_data_point(x, i, self._lfs))
        return self._matrix_from_row_data(labels)
