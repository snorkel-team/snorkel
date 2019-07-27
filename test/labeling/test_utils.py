import unittest

import numpy as np
import pandas as pd

from snorkel.labeling.utils import filter_unlabeled


class TestAnalysis(unittest.TestCase):
    def test_filter_unlabeled(self) -> None:
        X = pd.DataFrame([["x", 1], ["y", 2], ["z", 3]], columns=["A", "B"])
        y = np.array([[0.4, 0.3, 0.3], [1.0, 0.0, 0.0], [0.2, 0.3, 0.5]])
        L = np.array([[0, 1, -1], [-1, -1, -1], [1, 1, 0]])
        X_filtered, y_filtered = filter_unlabeled(X, y, L)
        np.array_equal(X_filtered.values, np.array([["x", 1], ["z", 3]]))
        np.testing.assert_array_almost_equal(
            y_filtered, np.array([[0.4, 0.3, 0.3], [0.2, 0.3, 0.5]])
        )
