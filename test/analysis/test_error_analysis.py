import unittest

import numpy as np

from snorkel.analysis.error_analysis import error_buckets


class ErrorAnalysisTest(unittest.TestCase):
    def test_error_buckets(self) -> None:
        golds = np.array([1, 2, 3, 1, 2, 3])
        preds = np.array([[2], [1], [3], [1], [1], [3]])
        buckets = error_buckets(golds, preds)
        expected_buckets = {(2, 1): [0], (1, 2): [1, 4], (3, 3): [2, 5], (1, 1): [3]}
        self.assertEqual(buckets, expected_buckets)

        preds_1d = np.array([2, 1, 3, 1, 1, 3])
        buckets = error_buckets(golds, preds_1d)
        self.assertEqual(buckets, expected_buckets)


if __name__ == "__main__":
    unittest.main()
