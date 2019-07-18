import unittest

import numpy as np
from scipy.sparse import csr_matrix

from snorkel.labeling.model.baselines import (
    MajorityClassVoter,
    MajorityLabelVoter,
    RandomVoter,
)


class BaselineModelTest(unittest.TestCase):
    def test_random_vote(self):
        L = csr_matrix([[1, 2, 1], [0, 4, 3], [3, 0, 0], [1, 2, 2]])
        rand_voter = RandomVoter()
        Y_p = rand_voter.predict_proba(L)
        self.assertLessEqual(Y_p.max(), 1.0)
        self.assertGreaterEqual(Y_p.min(), 0.0)
        np.testing.assert_array_almost_equal(
            np.sum(Y_p, axis=1), np.ones(np.shape(L)[0])
        )

    def test_majority_class_vote(self):
        L = csr_matrix([[1, 2, 1], [2, 2, 1], [2, 2, 1], [0, 0, 2]])
        mc_voter = MajorityClassVoter()
        mc_voter.train_model(balance=[0.8, 0.2])
        Y_p = mc_voter.predict_proba(L)

        Y_p_true = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(Y_p, Y_p_true)

    def test_majority_label_vote(self):
        L = csr_matrix([[1, 2, 1], [1, 2, 1], [2, 1, 1], [0, 0, 2]])
        ml_voter = MajorityLabelVoter()
        Y_p = ml_voter.predict_proba(L)

        Y_p_true = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(Y_p, Y_p_true)


if __name__ == "__main__":
    unittest.main()
