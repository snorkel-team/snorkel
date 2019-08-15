import unittest
from types import SimpleNamespace
from typing import List

from snorkel.preprocess import preprocessor
from snorkel.slicing import SFApplier, slicing_function
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


@slicing_function()
def f(x: DataPoint) -> int:
    return x.num > 42


@slicing_function(pre=[square])
def fp(x: DataPoint) -> int:
    return x.num_squared > 42


@slicing_function(resources=dict(db=[3, 6, 9]))
def g(x: DataPoint, db: List[int]) -> int:
    return x.num in db


DATA = [3, 43, 12, 9, 3]
S_EXPECTED = {"f": [0, 1, 0, 0, 0], "g": [1, 0, 0, 1, 1]}
S_PREPROCESS_EXPECTED = {"f": [0, 1, 0, 0, 0], "fp": [0, 1, 1, 1, 0]}


class TestSFApplier(unittest.TestCase):
    def test_sf_applier(self) -> None:
        data_points = [SimpleNamespace(num=num) for num in DATA]
        applier = SFApplier([f, g])
        S = applier.apply(data_points, progress_bar=False)
        self.assertEqual(S["f"].tolist(), S_EXPECTED["f"])
        self.assertEqual(S["g"].tolist(), S_EXPECTED["g"])
        S = applier.apply(data_points, progress_bar=True)
        self.assertEqual(S["f"].tolist(), S_EXPECTED["f"])
        self.assertEqual(S["g"].tolist(), S_EXPECTED["g"])

    def test_sf_applier_preprocessor(self) -> None:
        data_points = [SimpleNamespace(num=num) for num in DATA]
        applier = SFApplier([f, fp])
        S = applier.apply(data_points, progress_bar=False)
        self.assertEqual(S["f"].tolist(), S_PREPROCESS_EXPECTED["f"])
        self.assertEqual(S["fp"].tolist(), S_PREPROCESS_EXPECTED["fp"])
