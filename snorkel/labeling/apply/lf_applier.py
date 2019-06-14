from typing import Any, List

import scipy.sparse as sparse
from tqdm import tqdm

from snorkel.labeling.lf import LabelingFunction
from snorkel.types import DataPoints


class BaseLFApplier:
    def __init__(self, lfs: List[LabelingFunction]) -> None:
        self._lfs = lfs

    def apply(self, data_points: Any, *args: Any, **kwargs: Any) -> sparse.csr_matrix:
        raise NotImplementedError


class LFApplier(BaseLFApplier):
    def apply(self, data_points: DataPoints) -> sparse.csr_matrix:  # type: ignore
        L = sparse.lil_matrix((len(data_points), len(self._lfs)), dtype=int)
        for i, x in tqdm(enumerate(data_points)):
            for j, lf in enumerate(self._lfs):
                y = lf(x)
                if y != 0:
                    L[i, j] = y
        return L.tocsr()
