import unittest

import numpy as np
import pandas as pd

from snorkel.labeling import filter_unlabeled_dataframe


class TestAnalysis(unittest.TestCase):
    def test_filter_unlabeled_dataframe(self) -> None:
        X = pd.DataFrame(dict(A=["x", "y", "z"], B=[1, 2, 3]))
        y = np.array(
            [[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0], [0.2, 0.3, 0.5, 0.0]]
        )
        L = np.array([[0, 1, -1], [-1, -1, -1], [1, 1, 0]])
        X_filtered, y_filtered = filter_unlabeled_dataframe(X, y, L)
        np.array_equal(X_filtered.values, np.array([["x", 1], ["z", 3]]))
        np.testing.assert_array_almost_equal(
            y_filtered, np.array([[0.25, 0.25, 0.25, 0.25], [0.2, 0.3, 0.5, 0.0]])
        )
