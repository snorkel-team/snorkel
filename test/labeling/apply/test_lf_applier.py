import unittest
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from snorkel.labeling.apply import LFApplier, PandasLFApplier
from snorkel.labeling.lf import labeling_function
from snorkel.labeling.preprocess import Preprocessor
from snorkel.types import DataPoint, FieldMap


class SquarePreprocessor(Preprocessor):
    def __init__(self, x_field: str, squared_x_field: str) -> None:
        super().__init__(dict(x=x_field), dict(x=squared_x_field))

    def preprocess(self, x: float) -> FieldMap:  # type: ignore
        return dict(x=x ** 2)


square = SquarePreprocessor(x_field="a", squared_x_field="a")


@labeling_function()
def f(x: DataPoint) -> int:
    return 1 if x.a > 42 else 0


@labeling_function(preprocessors=[square])
def fp(x: DataPoint) -> int:
    return 1 if x.a > 42 else 0


@labeling_function(resources=dict(db=[3, 6, 9]))
def g(x: DataPoint, db: List[int]) -> int:
    return 1 if x.a in db else 0


DATA = [3, 43, 12, 9]
L_EXPECTED = np.array([[0, 1], [1, 0], [0, 0], [0, 1]])
L_PREPROCESS_EXPECTED = np.array([[0, 0], [1, 1], [0, 1], [0, 1]])


class TestLFApplier(unittest.TestCase):
    def test_lf_applier(self) -> None:
        data_points = [SimpleNamespace(a=a) for a in DATA]
        applier = LFApplier([f, g])
        L = applier.apply(data_points)
        np.testing.assert_equal(L.toarray(), L_EXPECTED)

    def test_lf_applier_preprocessor(self) -> None:
        data_points = [SimpleNamespace(a=a) for a in DATA]
        applier = LFApplier([f, fp])
        L = applier.apply(data_points)
        np.testing.assert_equal(L.toarray(), L_PREPROCESS_EXPECTED)

    def test_lf_applier_pandas(self) -> None:
        df = pd.DataFrame(dict(a=DATA))
        applier = PandasLFApplier([f, g])
        L = applier.apply(df)
        np.testing.assert_equal(L.toarray(), L_EXPECTED)

    def test_lf_applier_pandas_preprocessor(self) -> None:
        df = pd.DataFrame(dict(a=DATA))
        applier = PandasLFApplier([f, fp])
        L = applier.apply(df)
        np.testing.assert_equal(L.toarray(), L_PREPROCESS_EXPECTED)

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
