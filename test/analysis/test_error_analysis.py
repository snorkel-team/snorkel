import unittest

import numpy as np

from snorkel.analysis.error_analysis import label_buckets


class ErrorAnalysisTest(unittest.TestCase):
    def test_label_buckets(self) -> None:
        y1 = np.array([[2], [1], [3], [1], [1], [3]])
        y2 = np.array([1, 2, 3, 1, 2, 3])
        buckets = label_buckets(y1, y2)
        expected_buckets = {(2, 1): [0], (1, 2): [1, 4], (3, 3): [2, 5], (1, 1): [3]}
        expected_buckets = {k: np.array(v) for k, v in expected_buckets.items()}
        np.testing.assert_equal(buckets, expected_buckets)

        y1_1d = np.array([2, 1, 3, 1, 1, 3])
        buckets = label_buckets(y1_1d, y2)
        np.testing.assert_equal(buckets, expected_buckets)

        y3 = np.array([[3], [2], [1], [1], [2], [3]])
        buckets = label_buckets(y1, y2, y3)
        expected_buckets = {
            (2, 1, 3): [0],
            (1, 2, 2): [1, 4],
            (3, 3, 1): [2],
            (1, 1, 1): [3],
            (3, 3, 3): [5],
        }
        expected_buckets = {k: np.array(v) for k, v in expected_buckets.items()}
        np.testing.assert_equal(buckets, expected_buckets)


if __name__ == "__main__":
    unittest.main()
