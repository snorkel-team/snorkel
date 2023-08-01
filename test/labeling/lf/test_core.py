import pickle
import unittest
from types import SimpleNamespace
from typing import List

from snorkel.labeling import LabelingFunction, labeling_function
from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint


@preprocessor()
def square(x: DataPoint) -> DataPoint:
    x.num = x.num ** 2
    return x


@preprocessor()
def returns_none(x: DataPoint) -> DataPoint:
    return None


def f(x: DataPoint) -> int:
    return 0 if x.num > 42 else -1


def g(x: DataPoint, db: List[int]) -> int:
    return 0 if x.num in db else -1


class TestLabelingFunction(unittest.TestCase):
    def _run_lf(self, lf: LabelingFunction) -> None:
        x_43 = SimpleNamespace(num=43)
        x_19 = SimpleNamespace(num=19)
        self.assertEqual(lf(x_43), 0)
        self.assertEqual(lf(x_19), -1)

    def test_labeling_function(self) -> None:
        lf = LabelingFunction(name="my_lf", f=f)
        self._run_lf(lf)

    def test_labeling_function_resources(self) -> None:
        db = [3, 6, 43]
        lf = LabelingFunction(name="my_lf", f=g, resources=dict(db=db))
        self._run_lf(lf)

    def test_labeling_function_preprocessor(self) -> None:
        lf = LabelingFunction(name="my_lf", f=f, pre=[square, square])
        x_43 = SimpleNamespace(num=43)
        x_6 = SimpleNamespace(num=6)
        x_2 = SimpleNamespace(num=2)
        self.assertEqual(lf(x_43), 0)
        self.assertEqual(lf(x_6), 0)
        self.assertEqual(lf(x_2), -1)

    def test_labeling_function_returns_none(self) -> None:
        lf = LabelingFunction(name="my_lf", f=f, pre=[square, returns_none])
        x_43 = SimpleNamespace(num=43)
        with self.assertRaises(ValueError):
            lf(x_43)

    def test_labeling_function_serialize(self) -> None:
        db = [3, 6, 43]
        lf = LabelingFunction(name="my_lf", f=g, resources=dict(db=db))
        lf_load = pickle.loads(pickle.dumps(lf))
        self._run_lf(lf_load)

    def test_labeling_function_decorator(self) -> None:
        @labeling_function()
        def lf(x: DataPoint) -> int:
            return 0 if x.num > 42 else -1

        self.assertIsInstance(lf, LabelingFunction)
        self.assertEqual(lf.name, "lf")
        self._run_lf(lf)

    def test_labeling_function_decorator_args(self) -> None:
        db = [3, 6, 43 ** 2]

        @labeling_function(name="my_lf", resources=dict(db=db), pre=[square])
        def lf(x: DataPoint, db: List[int]) -> int:
            return 0 if x.num in db else -1

        self.assertIsInstance(lf, LabelingFunction)
        self.assertEqual(lf.name, "my_lf")
        self._run_lf(lf)

    def test_labeling_function_decorator_no_parens(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing parentheses"):

            @labeling_function
            def lf(x: DataPoint) -> int:
                return 0 if x.num > 42 else -1
