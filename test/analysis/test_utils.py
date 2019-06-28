import unittest

import numpy as np

from snorkel.analysis.utils import pred_to_prob, prob_to_pred

PROBS = np.array([[0.1, 0.9], [0.7, 0.3]])
PREDS = np.array([2, 1])
PREDS_SOFT = np.array([[0, 1], [1, 0]])


class MetricsTest(unittest.TestCase):
    def test_pred_to_prob(self):
        np.testing.assert_array_equal(pred_to_prob(PREDS, 2), PREDS_SOFT)

    def test_prob_to_pred(self):
        np.testing.assert_array_equal(prob_to_pred(PROBS), PREDS)


if __name__ == "__main__":
    unittest.main()
