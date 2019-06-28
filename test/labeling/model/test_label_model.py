import unittest

import numpy as np

# from snorkel.labeling.model.baselines import MajorityLabelVoter
from snorkel.labeling.model.label_model import LabelModel

# from .synthetic import SingleTaskTreeDepsGenerator


class LabelModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_iters = 1
        cls.n = 10000
        cls.m = 10
        cls.k = 2

    def test_model_init(self):
        label_model = LabelModel(k=2, verbose=False)

        # Test class balance
        Y_dev = np.array([1, 1, 2, 2, 1, 1, 1, 1, 2, 2])
        label_model._set_class_balance(class_balance=None, Y_dev=Y_dev)
        self.assertTrue((label_model.p == np.array([0.6, 0.4])).all())

        # Test dimension constants
        L = np.array([[1, 1, 2, 1], [2, 2, 1, 2], [1, 1, 1, 2]]).T
        label_model._set_constants(L)
        self.assertEqual(label_model.n, 4)
        self.assertEqual(label_model.m, 3)

    def test_generate_O(self):
        label_model = LabelModel(k=2, verbose=False)

        # Test inverse) from L
        L = np.array([[1, 1, 2, 1], [2, 2, 1, 2], [1, 1, 1, 2]]).T
        label_model._set_constants(L)
        label_model._set_dependencies(deps=[])
        label_model._generate_O(L)

        true_O = np.array(
            [
                [0.75, 0.00, 0.00, 0.75, 0.50, 0.25],
                [0.00, 0.25, 0.25, 0.00, 0.25, 0.00],
                [0.00, 0.25, 0.25, 0.00, 0.25, 0.00],
                [0.75, 0.00, 0.00, 0.75, 0.50, 0.25],
                [0.50, 0.25, 0.25, 0.50, 0.75, 0.00],
                [0.25, 0.00, 0.00, 0.25, 0.00, 0.25],
            ]
        )
        self.assertTrue((label_model.O.numpy() == true_O).all())

    def test_augmented_L_construction(self):
        # 5 LFs: a triangle, a connected edge to it, and a singleton source
        n = 3
        m = 5
        k = 2
        E = [(0, 1), (1, 2), (2, 0), (0, 3)]
        L = np.array([[1, 1, 1, 2, 1], [1, 2, 2, 1, 0], [1, 1, 1, 1, 0]])
        lm = LabelModel(k=k, verbose=False)
        lm._set_constants(L)
        lm._set_dependencies(E)
        L_aug = lm._get_augmented_label_matrix(L, higher_order=True)

        # Should have 22 columns:
        # - 5 * 2 = 10 for the sources
        # - 8 + 4 for the 3- and 2-clique resp. --> = 22
        self.assertEqual(L_aug.shape, (3, 22))

        # Same as above but minus 2 abstains = 19 total nonzero entries
        self.assertEqual(L_aug.sum(), 19)

        # Next, check the singleton entries
        for i in range(n):
            for j in range(m):
                if L[i, j] > 0:
                    self.assertEqual(L_aug[i, j * k + L[i, j] - 1], 1)

        # Finally, check the clique entries
        # Triangle clique
        self.assertEqual(len(lm.c_tree.node[1]["members"]), 3)
        j = lm.c_tree.node[1]["start_index"]
        self.assertEqual(L_aug[0, j], 1)
        self.assertEqual(L_aug[1, j + 3], 1)
        self.assertEqual(L_aug[2, j], 1)
        # Binary clique
        self.assertEqual(len(lm.c_tree.node[2]["members"]), 2)
        j = lm.c_tree.node[2]["start_index"]
        self.assertEqual(L_aug[0, j + 1], 1)
        self.assertEqual(L_aug[1, j], 1)
        self.assertEqual(L_aug[2, j], 1)

    # def test_no_deps(self):
    #     for seed in range(self.n_iters):
    #         np.random.seed(seed)
    #         data = SingleTaskTreeDepsGenerator(self.n, self.m, k=self.k, edge_prob=0.0)
    #         self._test_label_model(data)

    # def test_with_deps(self):
    #     for seed in range(self.n_iters):
    #         np.random.seed(seed)
    #         data = SingleTaskTreeDepsGenerator(self.n, self.m, k=self.k, edge_prob=1.0)
    #         self._test_label_model(data, test_acc=False)

    # def _test_label_model(self, data, test_acc=True):
    #     label_model = LabelModel(k=data.k, verbose=False)
    #     label_model.train_model(
    #         data.L,
    #         deps=data.E,
    #         class_balance=data.p,
    #         n_epochs=1000,
    #         log_train_every=200,
    #     )

    #     # Test parameter estimation error
    #     c_probs_est = label_model.get_conditional_probs()
    #     err = np.mean(np.abs(data.c_probs - c_probs_est))
    #     self.assertLess(err, 0.025)

    #     # Test label prediction accuracy
    #     if test_acc:
    #         score = label_model.score((data.L, data.Y), verbose=False)
    #         self.assertGreater(score, 0.95)

    #         # Test against baseline
    #         mv = MajorityLabelVoter()
    #         mv_score = mv.score((data.L, data.Y), verbose=False)
    #         self.assertGreater(score, mv_score)


if __name__ == "__main__":
    unittest.main()
