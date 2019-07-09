import unittest

import numpy as np

from snorkel.analysis.utils import (
    arraylike_to_numpy,
    convert_labels,
    filter_labels,
    preds_to_probs,
    probs_to_preds,
)

PROBS = np.array([[0.1, 0.9], [0.7, 0.3]])
PREDS = np.array([2, 1])
PREDS_ROUND = np.array([[0, 1], [1, 0]])


class MetricsTest(unittest.TestCase):
    def test_bad_input(self):
        not_array = "112200"
        Y_nparray = np.ones((3,))
        with self.assertRaisesRegex(ValueError, "Input could not be converted"):
            arraylike_to_numpy(not_array)

        with self.assertRaisesRegex(ValueError, "Unrecognized label data type"):
            convert_labels(Y=not_array, source="categorical", target="plusminus")

        self.assertIsNone(
            convert_labels(Y=None, source="categorical", target="plusminus")
        )
        np.testing.assert_array_equal(
            convert_labels(Y=Y_nparray, source="categorical", target="plusminus"),
            np.array([1, 1, 1]),
        )

    def test_pred_to_prob(self):
        np.testing.assert_array_equal(preds_to_probs(PREDS, 2), PREDS_ROUND)

    def test_prob_to_pred(self):
        np.testing.assert_array_equal(probs_to_preds(PROBS), PREDS)

    def test_filter_labels(self):
        golds = np.array([0, 1, 1, 2, 2])
        preds = np.array([1, 1, 2, 2, 0])
        filtered = filter_labels(
            label_dict={"golds": golds, "preds": preds},
            filter_dict={"golds": [0], "preds": [0]},
        )
        np.testing.assert_array_equal(filtered["golds"], np.array([1, 1, 2]))
        np.testing.assert_array_equal(filtered["preds"], np.array([1, 2, 2]))

    def test_filter_labels_probs(self):
        golds = np.array([0, 1, 1, 2, 2])
        preds = np.array([1, 1, 2, 2, 0])
        probs = np.array([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8], [0.5, 0.5]])
        filtered = filter_labels(
            label_dict={"golds": golds, "preds": preds, "probs": probs},
            filter_dict={"golds": [0], "preds": [0]},
        )
        np.testing.assert_array_equal(filtered["golds"], np.array([1, 1, 2]))
        np.testing.assert_array_equal(filtered["preds"], np.array([1, 2, 2]))


if __name__ == "__main__":
    unittest.main()
