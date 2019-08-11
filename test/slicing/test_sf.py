import unittest
from types import SimpleNamespace

from snorkel.slicing import SlicingFunction, slicing_function


class TestSlicingFunction(unittest.TestCase):
    def _run_sf(self, sf: SlicingFunction) -> None:
        x_43 = SimpleNamespace(num=43)
        x_19 = SimpleNamespace(num=19)
        self.assertEqual(sf(x_43), True)
        self.assertEqual(sf(x_19), False)

    def _run_sf_raise(self, sf: SlicingFunction) -> None:
        x_none = SimpleNamespace(num=None)
        with self.assertRaises(TypeError):
            sf(x_none)

    def _run_sf_no_raise(self, sf: SlicingFunction) -> None:
        x_none = SimpleNamespace(num=None)
        self.assertEqual(sf(x_none), -1)

    def test_slicing_function_decorator(self) -> None:
        @slicing_function()
        def sf(x) -> int:
            return x.num > 42

        self.assertIsInstance(sf, SlicingFunction)
        self.assertEqual(sf.name, "sf")
        self._run_sf(sf)
        self._run_sf_raise(sf)
