import unittest

import numpy as np

from snorkel.analysis.utils import pred_to_prob, prob_to_pred

PROB = np.array([[0.1, 0.9], [0.7, 0.3]])
PRED = np.array([2, 1])
PRED_SOFT = np.array([[0, 1], [1, 0]])


class MetricsTest(unittest.TestCase):
    def test_pred_to_prob(self):
        np.testing.assert_array_equal(pred_to_prob(PRED, 2), PRED_SOFT)

    def test_prob_to_pred(self):
        np.testing.assert_array_equal(prob_to_pred(PROB), PRED)


if __name__ == "__main__":
    unittest.main()
