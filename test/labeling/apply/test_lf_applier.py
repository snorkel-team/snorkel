import unittest
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from snorkel.labeling.apply import LFApplier, PandasLFApplier
from snorkel.labeling.lf import labeling_function
from snorkel.labeling.preprocess import preprocessor
from snorkel.labeling.preprocess.nlp import SpacyPreprocessor
from snorkel.types import DataPoint, FieldMap


@preprocessor
def square(num: float) -> FieldMap:
    return dict(num_squared=num ** 2)


spacy = SpacyPreprocessor(text_field="text", doc_field="doc")


@labeling_function()
def f(x: DataPoint) -> int:
    return 1 if x.num > 42 else 0


@labeling_function(preprocessors=[square])
def fp(x: DataPoint) -> int:
    return 1 if x.num_squared > 42 else 0


@labeling_function(resources=dict(db=[3, 6, 9]))
def g(x: DataPoint, db: List[int]) -> int:
    return 1 if x.num in db else 0


@labeling_function(preprocessors=[spacy])
def first_is_name(x: DataPoint) -> int:
    return 1 if x.doc[0].pos_ == "PROPN" else 0


@labeling_function(preprocessors=[spacy])
def has_verb(x: DataPoint) -> int:
    return 1 if sum(t.pos_ == "VERB" for t in x.doc) > 0 else 0


DATA = [3, 43, 12, 9]
L_EXPECTED = np.array([[0, 1], [1, 0], [0, 0], [0, 1]])
L_PREPROCESS_EXPECTED = np.array([[0, 0], [1, 1], [0, 1], [0, 1]])

TEXT_DATA = ["Jane", "Jane plays soccer."]
L_TEXT_EXPECTED = np.array([[1, 0], [1, 1]])


class TestLFApplier(unittest.TestCase):
    def test_lf_applier(self) -> None:
        data_points = [SimpleNamespace(num=num) for num in DATA]
        applier = LFApplier([f, g])
        L = applier.apply(data_points)
        np.testing.assert_equal(L.toarray(), L_EXPECTED)

    def test_lf_applier_preprocessor(self) -> None:
        data_points = [SimpleNamespace(num=num) for num in DATA]
        applier = LFApplier([f, fp])
        L = applier.apply(data_points)
        np.testing.assert_equal(L.toarray(), L_PREPROCESS_EXPECTED)

    def test_lf_applier_pandas(self) -> None:
        df = pd.DataFrame(dict(num=DATA))
        applier = PandasLFApplier([f, g])
        L = applier.apply(df)
        np.testing.assert_equal(L.toarray(), L_EXPECTED)

    def test_lf_applier_pandas_preprocessor(self) -> None:
        df = pd.DataFrame(dict(num=DATA))
        applier = PandasLFApplier([f, fp])
        L = applier.apply(df)
        np.testing.assert_equal(L.toarray(), L_PREPROCESS_EXPECTED)

    def test_lf_applier_pandas_spacy_preprocessor(self) -> None:
        df = pd.DataFrame(dict(text=TEXT_DATA))
        applier = PandasLFApplier([first_is_name, has_verb])
        L = applier.apply(df)
        np.testing.assert_equal(L.toarray(), L_TEXT_EXPECTED)

    def test_lf_applier_pandas_parquet(self) -> None:
        table_write = pa.Table.from_pandas(pd.DataFrame(dict(num=DATA)))
        stream = pa.BufferOutputStream()
        pq.write_table(table_write, stream)
        buffer = stream.getvalue()
        reader = pa.BufferReader(buffer)
        table = pq.read_table(reader)
        applier = PandasLFApplier([f, g])
        L = applier.apply(table.to_pandas())
        np.testing.assert_equal(L.toarray(), L_EXPECTED)
