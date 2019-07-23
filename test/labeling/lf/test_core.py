import pickle
import unittest
from types import SimpleNamespace

from snorkel.labeling.lf import LabelingFunction, labeling_function
from snorkel.labeling.preprocess import preprocessor
from snorkel.types import DataPoint


@preprocessor()
def square(x: DataPoint) -> DataPoint:
    x.num = x.num ** 2
    return x


@preprocessor()
def returns_none(x: DataPoint) -> DataPoint:
    return None


def f(x):
    if x.num > 42:
        return 0
    return None


def g(x, db):
    if x.num in db:
        return 0
    return None


class TestLabelingFunction(unittest.TestCase):
    def _run_lf(self, lf: LabelingFunction) -> None:
        x_43 = SimpleNamespace(num=43)
        x_19 = SimpleNamespace(num=19)
        self.assertEqual(lf(x_43), 0)
        self.assertIsNone(lf(x_19))

    def _run_lf_raise(self, lf: LabelingFunction) -> None:
        x_none = SimpleNamespace(num=None)
        with self.assertRaises(TypeError):
            lf(x_none)

    def _run_lf_no_raise(self, lf: LabelingFunction) -> None:
        x_none = SimpleNamespace(num=None)
        self.assertIsNone(lf(x_none))

    def test_labeling_function(self) -> None:
        lf = LabelingFunction(name="my_lf", f=f)
        self._run_lf(lf)
        self._run_lf_raise(lf)

    def test_labeling_function_fault_tolerant(self) -> None:
        lf = LabelingFunction(name="my_lf", f=f, fault_tolerant=True)
        self._run_lf(lf)
        self._run_lf_no_raise(lf)

    def test_labeling_function_resources(self) -> None:
        db = [3, 6, 43]
        lf = LabelingFunction(name="my_lf", f=g, resources=dict(db=db))
        self._run_lf(lf)
        self._run_lf_no_raise(lf)

    def test_labeling_function_preprocessor(self) -> None:
        lf = LabelingFunction(name="my_lf", f=f, preprocessors=[square, square])
        x_43 = SimpleNamespace(num=43)
        x_6 = SimpleNamespace(num=6)
        x_2 = SimpleNamespace(num=2)
        self.assertEqual(lf(x_43), 0)
        self.assertEqual(lf(x_6), 0)
        self.assertIsNone(lf(x_2))

    def test_labeling_function_returns_none(self) -> None:
        lf = LabelingFunction(name="my_lf", f=f, preprocessors=[square, returns_none])
        x_43 = SimpleNamespace(num=43)
        with self.assertRaises(ValueError):
            lf(x_43)

    def test_labeling_function_serialize(self) -> None:
        db = [3, 6, 43]
        lf = LabelingFunction(name="my_lf", f=g, resources=dict(db=db))
        lf_load = pickle.loads(pickle.dumps(lf))
        self._run_lf(lf_load)
        self._run_lf_no_raise(lf_load)

    def test_labeling_function_decorator(self) -> None:
        @labeling_function()
        def lf(x):
            if x.num > 42:
                return 0
            return None

        self.assertIsInstance(lf, LabelingFunction)
        self.assertEqual(lf.name, "lf")
        self._run_lf(lf)
        self._run_lf_raise(lf)

    def test_labeling_function_decorator_args(self) -> None:
        @labeling_function(name="my_lf", fault_tolerant=True)
        def lf(x):
            if x.num > 42:
                return 0
            return None

        self.assertIsInstance(lf, LabelingFunction)
        self.assertEqual(lf.name, "my_lf")
        self._run_lf(lf)
        self._run_lf_no_raise(lf)
