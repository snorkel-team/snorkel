from typing import List, Tuple

import scipy.sparse as sparse
from pandas import DataFrame
from tqdm import tqdm

from snorkel.types import DataPoint

from .lf_applier import BaseLFApplier

PandasRowData = List[Tuple[int, int]]


class PandasLFApplier(BaseLFApplier):
    def apply(self, df: DataFrame) -> sparse.csr_matrix:  # type: ignore
        def apply_fn(x: DataPoint) -> PandasRowData:
            labels = []
            for j, lf in enumerate(self._lfs):
                y = lf(x)
                if y != 0:
                    labels.append((j, y))
            return labels

        tqdm.pandas()
        labels = df.progress_apply(apply_fn, axis=1)
        L = sparse.lil_matrix((len(df), len(self._lfs)), dtype=int)
        for i, row_labels in enumerate(labels):
            for j, y in row_labels:
                L[i, j] = y
        return L.tocsr()
