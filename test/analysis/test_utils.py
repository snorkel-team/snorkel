import unittest

import numpy as np

from snorkel.analysis.utils import (
    break_ties,
    convert_labels,
    filter_labels,
    preds_to_probs,
    probs_to_preds,
    to_int_label_array,
)

PROBS = np.array([[0.1, 0.9], [0.7, 0.3]])
PREDS = np.array([2, 1])
PREDS_ROUND = np.array([[0, 1], [1, 0]])


class MetricsTest(unittest.TestCase):
    def test_inputs(self):
        Y_nparray = np.ones((3,))
        self.assertIsNone(
            convert_labels(Y=None, source="categorical", target="plusminus")
        )
        np.testing.assert_array_equal(
            convert_labels(Y=Y_nparray, source="categorical", target="plusminus"),
            np.array([1, 1, 1]),
        )

    def test_to_int_label_array(self):
        X = np.array([[1], [0], [2.0]])
        Y_expected = np.array([1, 0, 2])
        Y = to_int_label_array(X, flatten_vector=True)
        np.testing.assert_array_equal(Y, Y_expected)

        Y = to_int_label_array(X, flatten_vector=False)
        Y_expected = np.array([[1], [0], [2]])
        np.testing.assert_array_equal(Y, Y_expected)

        X = np.array([[1], [0], [2.1]])
        with self.assertRaisesRegex(ValueError, "non-integer value"):
            to_int_label_array(X)

        X = np.array([[1, 0], [0, 1]])
        with self.assertRaisesRegex(ValueError, "1d np.array"):
            to_int_label_array(X, flatten_vector=True)

    def test_pred_to_prob(self):
        np.testing.assert_array_equal(preds_to_probs(PREDS, 2), PREDS_ROUND)

    def test_break_ties(self):
        # abtains with ties
        probs = np.array([[0.33, 0.33, 0.33]])
        preds = break_ties(probs, tie_break_policy="abstain")
        true_preds = np.array([0.0])
        np.testing.assert_array_equal(preds, true_preds)

        # true random with ties
        probs = np.array([[0.33, 0.33, 0.33]])
        random_preds = []
        for seed in range(10):
            preds = break_ties(probs, tie_break_policy="true-random")
            random_preds.append(preds[0])

        # check predicted labels within range
        self.assertLessEqual(max(random_preds), 3)
        self.assertGreaterEqual(min(random_preds), 1)

        # deterministic random with ties
        probs = np.array(
            [[0.33, 0.33, 0.33], [0.0, 0.5, 0.5], [0.33, 0.33, 0.33], [0.5, 0.5, 0]]
        )
        random_preds = []
        for seed in range(10):
            preds = break_ties(probs, tie_break_policy="random")
            random_preds.append(preds)

        # check labels are same across seeds
        for i in range(len(random_preds) - 1):
            np.testing.assert_array_equal(random_preds[i], random_preds[i + 1])

        # check predicted labels within range (only one instance since should all be same)
        self.assertLessEqual(max(random_preds[0]), 3)
        self.assertGreaterEqual(min(random_preds[0]), 1)

        # check invalid policy
        with self.assertRaisesRegex(ValueError, "policy not recognized"):
            preds = break_ties(probs, tie_break_policy="negative")

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
