import unittest

import numpy as np

from snorkel.analysis.error_analysis import confusion_matrix, error_buckets


class ErrorAnalysisTest(unittest.TestCase):
    def test_error_buckets(self) -> None:
        golds = np.array([1, 2, 3, 1, 2, 3])
        preds = np.array([2, 1, 3, 1, 1, 3])
        buckets = error_buckets(golds, preds, X=None)
        self.assertEqual(
            buckets, {(2, 1): [0], (1, 2): [1, 4], (3, 3): [2, 5], (1, 1): [3]}
        )

    def test_confusion_matrix(self) -> None:
        preds = [0, 2, 2, 3, 1, 0, 1, 3]
        golds = [1, 2, 2, 0, 3, 0, 2, 3]

        mat = confusion_matrix(
            golds, preds, null_pred=False, null_gold=False, normalize=False
        )
        mat_expected = np.array([[0, 1, 1], [0, 2, 0], [0, 0, 1]])
        np.testing.assert_array_equal(mat, mat_expected)

        mat = confusion_matrix(
            golds, preds, null_pred=False, null_gold=False, normalize=True
        )
        mat_expected = np.array([[0, 1 / 8, 1 / 8], [0, 2 / 8, 0], [0, 0, 1 / 8]])
        np.testing.assert_array_equal(mat, mat_expected)

        mat = confusion_matrix(
            golds, preds, null_pred=True, null_gold=False, normalize=False
        )
        mat_expected = np.array([[1, 0, 0], [0, 1, 1], [0, 2, 0], [0, 0, 1]])
        np.testing.assert_array_equal(mat, mat_expected)

        mat = confusion_matrix(
            golds, preds, null_pred=True, null_gold=True, normalize=False
        )
        mat_expected = np.array(
            [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 2, 0], [1, 0, 0, 1]]
        )
        np.testing.assert_array_equal(mat, mat_expected)
