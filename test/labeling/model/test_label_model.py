import unittest

import numpy as np

from snorkel.labeling.model.baselines import MajorityLabelVoter
from snorkel.labeling.model.label_model import LabelModel

from .synthetic import SingleTaskTreeDepsGenerator


# TODO: Put in tests for LabelModel baselines again!
class LabelModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_iters = 1
        cls.n = 10000
        cls.m = 10
        cls.k = 2

    def _test_label_model(self, data, test_acc=True):
        label_model = LabelModel(k=data.k, verbose=False)
        label_model.train_model(
            data.L, class_balance=data.p, n_epochs=1000, log_train_every=200
        )

        # Test parameter estimation error
        c_probs_est = label_model.get_conditional_probs()
        err = np.mean(np.abs(data.c_probs - c_probs_est))
        self.assertLess(err, 0.025)

        # Test label prediction accuracy
        if test_acc:
            score = label_model.score((data.L, data.Y), verbose=False)
            self.assertGreater(score, 0.95)

            # Test against baseline
            mv = MajorityLabelVoter()
            mv_score = mv.score((data.L, data.Y), verbose=False)
            self.assertGreater(score, mv_score)

    def test_no_deps(self):
        for seed in range(self.n_iters):
            np.random.seed(seed)
            data = SingleTaskTreeDepsGenerator(self.n, self.m, k=self.k, edge_prob=0.0)
            self._test_label_model(data)

    def test_augmented_L_construction(self):
        # 5 LFs
        n = 3
        m = 5
        k = 2
        L = np.array([[1, 1, 1, 2, 1], [1, 2, 2, 1, 0], [1, 1, 1, 1, 0]])
        lm = LabelModel(k=k, verbose=False)
        lm._set_constants(L)
        lm._create_tree()
        L_aug = lm._get_augmented_label_matrix(L, higher_order=True)

        # Should have 10 columns:
        # - 5 * 2 = 10 for the sources
        self.assertEqual(L_aug.shape, (3, 10))

        # 13 total nonzero entries
        self.assertEqual(L_aug.sum(), 13)

        # Next, check the singleton entries
        for i in range(n):
            for j in range(m):
                if L[i, j] > 0:
                    self.assertEqual(L_aug[i, j * k + L[i, j] - 1], 1)

        # Finally, check the clique entries
        # Singleton clique 1
        self.assertEqual(len(lm.c_tree.node[1]["members"]), 1)
        j = lm.c_tree.node[1]["start_index"]
        self.assertEqual(L_aug[0, j], 1)

        # Singleton clique 2
        self.assertEqual(len(lm.c_tree.node[2]["members"]), 1)
        j = lm.c_tree.node[2]["start_index"]
        self.assertEqual(L_aug[0, j + 1], 0)


if __name__ == "__main__":
    unittest.main()
