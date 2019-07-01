import unittest

import numpy as np
import torch

from snorkel.labeling.model.label_model import LabelModel


class LabelModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_iters = 1
        cls.n = 10000
        cls.m = 10
        cls.k = 2

    def test_model_constants(self):
        label_model = LabelModel(k=2, verbose=False)

        # Test dimension constants
        L = np.array([[1, 2, 1], [1, 2, 1], [2, 1, 1], [1, 2, 2]])
        label_model._set_constants(L)
        self.assertEqual(label_model.n, 4)
        self.assertEqual(label_model.m, 3)

    def test_class_balance(self):
        label_model = LabelModel(k=2, verbose=False)

        # Test class balance
        Y_dev = np.array([1, 1, 2, 2, 1, 1, 1, 1, 2, 2])
        label_model._set_class_balance(class_balance=None, Y_dev=Y_dev)
        np.testing.assert_array_almost_equal(label_model.p, np.array([0.6, 0.4]))

    def test_generate_O(self):
        label_model = LabelModel(k=2, verbose=False)

        # Test inverse from L
        L = np.array([[1, 2, 1], [1, 2, 1], [2, 1, 1], [1, 2, 2]])
        label_model._set_constants(L)
        label_model._set_dependencies(deps=[])
        label_model._generate_O(L)

        true_O = np.array(
            [
                [3.0 / 4.0, 0.0, 0.0, 3.0 / 4.0, 1.0 / 2.0, 1.0 / 4.0],
                [0.0, 1.0 / 4.0, 1.0 / 4.0, 0.0, 1.0 / 4.0, 0.0],
                [0.0, 1.0 / 4.0, 1.0 / 4.0, 0.0, 1.0 / 4.0, 0.0],
                [3.0 / 4.0, 0.0, 0.0, 3.0 / 4.0, 1.0 / 2.0, 1.0 / 4.0],
                [1.0 / 2.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0, 3.0 / 4.0, 0.0],
                [1.0 / 4.0, 0.0, 0.0, 1.0 / 4.0, 0.0, 1.0 / 4.0],
            ]
        )
        np.testing.assert_array_almost_equal(label_model.O.numpy(), true_O)

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

    def test_conditional_probs(self):
        label_model = LabelModel(k=2, verbose=False)

        L = np.array([[1, 2, 1], [1, 2, 1]])
        label_model._set_constants(L)
        label_model._set_dependencies(deps=[])
        label_model._generate_O(L)
        label_model.inv_form = False
        label_model._set_class_balance(class_balance=[0.6, 0.4], Y_dev=None)
        label_model._init_params()

        probs = label_model.get_conditional_probs()
        self.assertLessEqual(probs.max(), 1.0)
        self.assertGreaterEqual(probs.min(), 0.0)

    def test_get_accuracy(self):
        label_model = LabelModel(k=2, verbose=False)
        probs = np.array(
            [
                [0.99, 0.01],
                [0.5, 0.5],
                [0.9, 0.9],
                [0.99, 0.01],
                [0.9, 0.9],
                [0.5, 0.75],
                [0.9, 0.9],
                [0.9, 0.1],
            ]
        )

        label_model.m = 2
        label_model.k = 2
        label_model.P = torch.Tensor([[0.5, 0.0], [0.0, 0.5]])
        accs = label_model.get_accuracies(probs=probs)
        np.testing.assert_array_almost_equal(accs, np.array([0.7, 0.825]))

    def test_build_mask(self):
        label_model = LabelModel(k=2, verbose=False)
        L = np.array([[1, 2, 1], [1, 2, 1]])
        label_model._set_constants(L)
        label_model._set_dependencies(deps=[])
        label_model._generate_O(L)
        label_model._build_mask()
        mask = label_model.mask.numpy()

        true_mask = np.array([[0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0]])

        np.testing.assert_array_equal(mask, true_mask)

        label_model._set_dependencies(deps=[(1, 2)])
        label_model._generate_O(L)
        label_model._build_mask()
        mask = label_model.mask.numpy()

        true_mask = np.array([[0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0]])

        np.testing.assert_array_equal(mask, true_mask)


if __name__ == "__main__":
    unittest.main()
