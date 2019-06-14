import unittest
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from snorkel.labeling.apply import LFApplier, PandasLFApplier
from snorkel.labeling.lf import labeling_function
from snorkel.types import DataPoint


@labeling_function()
def f(x: DataPoint) -> int:
    return 1 if x.a > 42 else 0


@labeling_function(resources=dict(db=[3, 6, 9]))
def g(x: DataPoint, db: List[int]) -> int:
    return 1 if x.a in db else 0


DATA = [3, 43, 12, 9]
L_EXPECTED = np.array([[0, 1], [1, 0], [0, 0], [0, 1]])


class TestLFApplier(unittest.TestCase):
    def test_lf_applier(self) -> None:
        data_points = [SimpleNamespace(a=a) for a in DATA]
        applier = LFApplier([f, g])
        L = applier.apply(data_points)
        np.testing.assert_equal(L.toarray(), L_EXPECTED)

    def test_lf_applier_pandas(self) -> None:
        df = pd.DataFrame(dict(a=DATA))
        applier = PandasLFApplier([f, g])
        L = applier.apply(df)
        np.testing.assert_equal(L.toarray(), L_EXPECTED)

    def test_lf_applier_pandas_parquet(self) -> None:
        table_write = pa.Table.from_pandas(pd.DataFrame(dict(a=DATA)))
        stream = pa.BufferOutputStream()
        pq.write_table(table_write, stream)
        buffer = stream.getvalue()
        reader = pa.BufferReader(buffer)
        table = pq.read_table(reader)
        applier = PandasLFApplier([f, g])
        L = applier.apply(table.to_pandas())
        np.testing.assert_equal(L.toarray(), L_EXPECTED)
