import pickle
import unittest
from types import SimpleNamespace
from typing import List

from snorkel.labeling.lf.nlp import NLPLabelingFunction, nlp_labeling_function
from snorkel.labeling.preprocess import preprocessor
from snorkel.types import DataPoint


@preprocessor()
def square(x: DataPoint) -> DataPoint:
    x.num = x.num ** 2
    return x


@preprocessor()
def returns_none(x: DataPoint) -> DataPoint:
    return None


def f(x: DataPoint) -> int:
    return 1 if x.num > 42 else 0


def g(x: DataPoint, db: List[int]) -> int:
    return 1 if x.num in db else 0


class TestNLPNLPLabelingFunction(unittest.TestCase):
    def _run_lf(self, lf: NLPLabelingFunction) -> None:
        x_43 = SimpleNamespace(num=43)
        x_19 = SimpleNamespace(num=19)
        self.assertEqual(lf(x_43), 1)
        self.assertEqual(lf(x_19), 0)

    def test_labeling_function(self) -> None:
        lf = NLPLabelingFunction(name="my_lf", f=f)
        self._run_lf(lf)

    def test_labeling_function_fault_tolerant(self) -> None:
        lf = NLPLabelingFunction(name="my_lf", f=f, fault_tolerant=True)
        self._run_lf(lf)

    def test_labeling_function_resources(self) -> None:
        db = [3, 6, 43]
        lf = NLPLabelingFunction(name="my_lf", f=g, resources=dict(db=db))
        self._run_lf(lf)

    def test_labeling_function_preprocessor(self) -> None:
        lf = NLPLabelingFunction(name="my_lf", f=f, preprocessors=[square, square])
        x_43 = SimpleNamespace(num=43)
        x_6 = SimpleNamespace(num=6)
        x_2 = SimpleNamespace(num=2)
        self.assertEqual(lf(x_43), 1)
        self.assertEqual(lf(x_6), 1)
        self.assertEqual(lf(x_2), 0)

    def test_labeling_function_returns_none(self) -> None:
        lf = NLPLabelingFunction(
            name="my_lf", f=f, preprocessors=[square, returns_none]
        )
        x_43 = SimpleNamespace(num=43)
        with self.assertRaises(ValueError):
            lf(x_43)

    def test_labeling_function_serialize(self) -> None:
        db = [3, 6, 43]
        lf = NLPLabelingFunction(name="my_lf", f=g, resources=dict(db=db))
        lf_load = pickle.loads(pickle.dumps(lf))
        self._run_lf(lf_load)

    def test_labeling_function_decorator(self) -> None:
        @nlp_labeling_function()
        def lf(x: DataPoint) -> int:
            return 1 if x.num > 42 else 0

        self.assertEqual(lf.name, "lf")
        self._run_lf(lf)
        self._run_lf_raise(lf)

    def test_labeling_function_decorator_args(self) -> None:
        @nlp_labeling_function(name="my_lf", fault_tolerant=True)
        def lf(x: DataPoint) -> int:
            return 1 if x.num > 42 else 0

        self.assertEqual(lf.name, "my_lf")
        self._run_lf(lf)
