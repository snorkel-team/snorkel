import unittest
from typing import List

import numpy as np
import pandas as pd
import pytest
from pyspark import SparkContext
from pyspark.sql import Row, SQLContext

from snorkel.labeling import labeling_function
from snorkel.labeling.apply.spark import SparkLFApplier
from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint


@preprocessor()
def square(x: Row) -> Row:
    return Row(num=x.num, num_squared=x.num ** 2)


@labeling_function()
def f(x: DataPoint) -> int:
    return 0 if x.num > 42 else -1


@labeling_function(pre=[square])
def fp(x: DataPoint) -> int:
    return 0 if x.num_squared > 42 else -1


@labeling_function(resources=dict(db=[3, 6, 9]))
def g(x: DataPoint, db: List[int]) -> int:
    return 0 if x.num in db else -1


@labeling_function()
def f_bad(x: DataPoint) -> int:
    return 0 if x.mum > 42 else -1


DATA = [3, 43, 12, 9, 3]
L_EXPECTED = np.array([[-1, 0], [0, -1], [-1, -1], [-1, 0], [-1, 0]])
L_EXPECTED_BAD = np.array([[-1, -1], [0, -1], [-1, -1], [-1, -1], [-1, -1]])
L_PREPROCESS_EXPECTED = np.array([[-1, -1], [0, 0], [-1, 0], [-1, 0], [-1, -1]])

TEXT_DATA = ["Jane", "Jane plays soccer.", "Jane plays soccer."]
L_TEXT_EXPECTED = np.array([[0, -1], [0, 0], [0, 0]])


class TestSparkApplier(unittest.TestCase):
    @pytest.mark.complex
    @pytest.mark.spark
    def test_lf_applier_spark(self) -> None:
        sc = SparkContext.getOrCreate()
        sql = SQLContext(sc)
        df = pd.DataFrame(dict(num=DATA))
        rdd = sql.createDataFrame(df).rdd
        applier = SparkLFApplier([f, g])
        L = applier.apply(rdd)
        np.testing.assert_equal(L, L_EXPECTED)

    @pytest.mark.complex
    @pytest.mark.spark
    def test_lf_applier_spark_fault(self) -> None:
        sc = SparkContext.getOrCreate()
        sql = SQLContext(sc)
        df = pd.DataFrame(dict(num=DATA))
        rdd = sql.createDataFrame(df).rdd
        applier = SparkLFApplier([f, f_bad])
        with self.assertRaises(Exception):
            applier.apply(rdd)
        L = applier.apply(rdd, fault_tolerant=True)
        np.testing.assert_equal(L, L_EXPECTED_BAD)

    @pytest.mark.complex
    @pytest.mark.spark
    def test_lf_applier_spark_preprocessor(self) -> None:
        sc = SparkContext.getOrCreate()
        sql = SQLContext(sc)
        df = pd.DataFrame(dict(num=DATA))
        rdd = sql.createDataFrame(df).rdd
        applier = SparkLFApplier([f, fp])
        L = applier.apply(rdd)
        np.testing.assert_equal(L, L_PREPROCESS_EXPECTED)

    @pytest.mark.complex
    @pytest.mark.spark
    def test_lf_applier_spark_preprocessor_memoized(self) -> None:
        sc = SparkContext.getOrCreate()
        sql = SQLContext(sc)

        @preprocessor(memoize=True)
        def square_memoize(x: DataPoint) -> DataPoint:
            return Row(num=x.num, num_squared=x.num ** 2)

        @labeling_function(pre=[square_memoize])
        def fp_memoized(x: DataPoint) -> int:
            return 0 if x.num_squared > 42 else -1

        df = pd.DataFrame(dict(num=DATA))
        rdd = sql.createDataFrame(df).rdd
        applier = SparkLFApplier([f, fp_memoized])
        L = applier.apply(rdd)
        np.testing.assert_equal(L, L_PREPROCESS_EXPECTED)
