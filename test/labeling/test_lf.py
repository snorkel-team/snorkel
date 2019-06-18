import pickle
import unittest
from types import SimpleNamespace
from typing import List

from snorkel.labeling.lf import LabelingFunction, labeling_function
from snorkel.labeling.preprocess import Preprocessor, PreprocessorMode
from snorkel.types import DataPoint, FieldMap


class SquarePreprocessor(Preprocessor):
    def __init__(self, x_field: str, squared_x_field: str) -> None:
        super().__init__(dict(x=x_field), dict(x=squared_x_field))

    def preprocess(self, x: float) -> FieldMap:  # type: ignore
        return dict(x=x ** 2)


def f(x: DataPoint) -> int:
    return 1 if x.a > 42 else 0


def g(x: DataPoint, db: List[int]) -> int:
    return 1 if x.a in db else 0


class TestLabelingFunctionCore(unittest.TestCase):
    def _run_lf(self, lf: LabelingFunction) -> None:
        x_43 = SimpleNamespace(a=43)
        x_19 = SimpleNamespace(a=19)
        self.assertEqual(lf(x_43), 1)
        self.assertEqual(lf(x_19), 0)

    def _run_lf_raise(self, lf: LabelingFunction) -> None:
        x_none = SimpleNamespace(a=None)
        with self.assertRaises(TypeError):
            lf(x_none)

    def _run_lf_no_raise(self, lf: LabelingFunction) -> None:
        x_none = SimpleNamespace(a=None)
        self.assertEqual(lf(x_none), 0)

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
        p = SquarePreprocessor(x_field="a", squared_x_field="a")
        lf = LabelingFunction(name="my_lf", f=f, preprocessors=[p, p])
        lf.set_preprocessor_mode(PreprocessorMode.NAMESPACE)
        x_43 = SimpleNamespace(a=43)
        x_6 = SimpleNamespace(a=6)
        x_2 = SimpleNamespace(a=2)
        self.assertEqual(lf(x_43), 1)
        self.assertEqual(lf(x_6), 1)
        self.assertEqual(lf(x_2), 0)

    def test_labeling_function_serialize(self) -> None:
        db = [3, 6, 43]
        lf = LabelingFunction(name="my_lf", f=g, resources=dict(db=db))
        lf_load = pickle.loads(pickle.dumps(lf))
        self._run_lf(lf_load)
        self._run_lf_no_raise(lf_load)

    def test_labeling_function_decorator(self) -> None:
        @labeling_function()
        def lf(x: DataPoint) -> int:
            return 1 if x.a > 42 else 0

        self.assertEqual(lf.name, "lf")
        self._run_lf(lf)
        self._run_lf_raise(lf)

    def test_labeling_function_decorator_args(self) -> None:
        @labeling_function(name="my_lf", fault_tolerant=True)
        def lf(x: DataPoint) -> int:
            return 1 if x.a > 42 else 0

        self.assertEqual(lf.name, "my_lf")
        self._run_lf(lf)
        self._run_lf_no_raise(lf)
