import unittest
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
from dask import dataframe as dd

from snorkel.labeling.apply import (
    DaskLFApplier,
    LFApplier,
    PandasLFApplier,
    PandasParallelLFApplier,
)
from snorkel.labeling.lf import labeling_function
from snorkel.labeling.preprocess import preprocessor
from snorkel.labeling.preprocess.nlp import SpacyPreprocessor
from snorkel.types import DataPoint


@preprocessor()
def square(x: DataPoint) -> DataPoint:
    x.num_squared = x.num ** 2
    return x


class SquareHitTracker:
    def __init__(self):
        self.n_hits = 0

    def __call__(self, x: float) -> float:
        self.n_hits += 1
        return x ** 2


@labeling_function()
def f(x: DataPoint) -> int:
    return 1 if x.num > 42 else 0


@labeling_function(preprocessors=[square])
def fp(x: DataPoint) -> int:
    return 1 if x.num_squared > 42 else 0


@labeling_function(resources=dict(db=[3, 6, 9]))
def g(x: DataPoint, db: List[int]) -> int:
    return 1 if x.num in db else 0


DATA = [3, 43, 12, 9, 3]
L_EXPECTED = np.array([[0, 1], [1, 0], [0, 0], [0, 1], [0, 1]])
L_PREPROCESS_EXPECTED = np.array([[0, 0], [1, 1], [0, 1], [0, 1], [0, 0]])

TEXT_DATA = ["Jane", "Jane plays soccer.", "Jane plays soccer."]
L_TEXT_EXPECTED = np.array([[1, 0], [1, 1], [1, 1]])


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

    def test_lf_applier_preprocessor_memoized(self) -> None:
        data_points = [SimpleNamespace(num=num) for num in DATA]
        square_hit_tracker = SquareHitTracker()

        @preprocessor(memoize=True)
        def square_memoize(x: DataPoint) -> DataPoint:
            x.num_squared = square_hit_tracker(x.num)
            return x

        @labeling_function(preprocessors=[square_memoize])
        def fp_memoized(x: DataPoint) -> int:
            return 1 if x.num_squared > 42 else 0

        applier = LFApplier([f, fp_memoized])
        L = applier.apply(data_points)
        np.testing.assert_equal(L.toarray(), L_PREPROCESS_EXPECTED)
        self.assertEqual(square_hit_tracker.n_hits, 4)


class TestPandasApplier(unittest.TestCase):
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

    def test_lf_applier_pandas_preprocessor_memoized(self) -> None:
        df = pd.DataFrame(dict(num=DATA))
        square_hit_tracker = SquareHitTracker()

        @preprocessor(memoize=True)
        def square_memoize(x: DataPoint) -> DataPoint:
            x.num_squared = square_hit_tracker(x.num)
            return x

        @labeling_function(preprocessors=[square_memoize])
        def fp_memoized(x: DataPoint) -> int:
            return 1 if x.num_squared > 42 else 0

        applier = PandasLFApplier([f, fp_memoized])
        L = applier.apply(df)
        np.testing.assert_equal(L.toarray(), L_PREPROCESS_EXPECTED)
        self.assertEqual(square_hit_tracker.n_hits, 4)

    def test_lf_applier_pandas_spacy_preprocessor(self) -> None:
        spacy = SpacyPreprocessor(text_field="text", doc_field="doc")

        @labeling_function(preprocessors=[spacy])
        def first_is_name(x: DataPoint) -> int:
            return 1 if x.doc[0].pos_ == "PROPN" else 0

        @labeling_function(preprocessors=[spacy])
        def has_verb(x: DataPoint) -> int:
            return 1 if sum(t.pos_ == "VERB" for t in x.doc) > 0 else 0

        df = pd.DataFrame(dict(text=TEXT_DATA))
        applier = PandasLFApplier([first_is_name, has_verb])
        L = applier.apply(df)
        np.testing.assert_equal(L.toarray(), L_TEXT_EXPECTED)

    def test_lf_applier_pandas_spacy_preprocessor_memoized(self) -> None:
        spacy = SpacyPreprocessor(text_field="text", doc_field="doc")
        spacy.memoize_outputs(True)

        @labeling_function(preprocessors=[spacy])
        def first_is_name(x: DataPoint) -> int:
            return 1 if x.doc[0].pos_ == "PROPN" else 0

        @labeling_function(preprocessors=[spacy])
        def has_verb(x: DataPoint) -> int:
            return 1 if sum(t.pos_ == "VERB" for t in x.doc) > 0 else 0

        df = pd.DataFrame(dict(text=TEXT_DATA))
        applier = PandasLFApplier([first_is_name, has_verb])
        L = applier.apply(df)
        np.testing.assert_equal(L.toarray(), L_TEXT_EXPECTED)
        self.assertEqual(len(spacy._cache), 2)


class TestDaskApplier(unittest.TestCase):
    def test_lf_applier_dask(self) -> None:
        df = pd.DataFrame(dict(num=DATA))
        df = dd.from_pandas(df, npartitions=2)
        applier = DaskLFApplier([f, g])
        L = applier.apply(df)
        np.testing.assert_equal(L.toarray(), L_EXPECTED)

    def test_lf_applier_dask_preprocessor(self) -> None:
        df = pd.DataFrame(dict(num=DATA))
        df = dd.from_pandas(df, npartitions=2)
        applier = DaskLFApplier([f, fp])
        L = applier.apply(df)
        np.testing.assert_equal(L.toarray(), L_PREPROCESS_EXPECTED)

    def test_lf_applier_dask_spacy_preprocessor(self) -> None:
        spacy = SpacyPreprocessor(text_field="text", doc_field="doc")

        @labeling_function(preprocessors=[spacy])
        def first_is_name(x: DataPoint) -> int:
            return 1 if x.doc[0].pos_ == "PROPN" else 0

        @labeling_function(preprocessors=[spacy])
        def has_verb(x: DataPoint) -> int:
            return 1 if sum(t.pos_ == "VERB" for t in x.doc) > 0 else 0

        df = pd.DataFrame(dict(text=TEXT_DATA))
        df = dd.from_pandas(df, npartitions=2)
        applier = DaskLFApplier([first_is_name, has_verb])
        L = applier.apply(df)
        np.testing.assert_equal(L.toarray(), L_TEXT_EXPECTED)

    def test_lf_applier_pandas_parallel(self) -> None:
        df = pd.DataFrame(dict(num=DATA))
        applier = PandasParallelLFApplier([f, g])
        L = applier.apply(df, n_parallel=2)
        np.testing.assert_equal(L.toarray(), L_EXPECTED)

    def test_lf_applier_pandas_parallel_raises(self) -> None:
        df = pd.DataFrame(dict(num=DATA))
        applier = PandasParallelLFApplier([f, g])
        with self.assertRaises(ValueError):
            applier.apply(df, n_parallel=1)
