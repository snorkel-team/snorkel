import unittest

import pandas as pd

from snorkel.slicing import slicing_function
from snorkel.slicing.monitor import slice_dataframe

DATA = [5, 10, 19, 22, 25]


@slicing_function()
def sf(x):
    return x.num < 20


class PandasSlicerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame(dict(num=DATA))

    def test_slice(self):
        self.assertEqual(len(self.df), 5)

        # Should return a subset
        sliced_df = slice_dataframe(self.df, sf)
        self.assertEqual(len(sliced_df), 3)
