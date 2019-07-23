import shutil
import tempfile
import unittest
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn

from snorkel.labeling.model.label_model import LabelModel
from snorkel.synthetic.synthetic_data import generate_simple_label_matrix


class LabelModelTest(unittest.TestCase):
    def _set_up_model(self, L: np.ndarray, class_balance: List[float] = [0.5, 0.5]):
        label_model = LabelModel(cardinality=2, verbose=False)
        L_aug = L + 1
        label_model._set_constants(L_aug)
        label_model._create_tree()
        label_model._generate_O(L_aug)
        label_model._build_mask()
        label_model._get_augmented_label_matrix(L_aug)
        label_model._set_class_balance(class_balance=class_balance, Y_dev=None)
        label_model._init_params()
        return label_model

    def test_L_form(self):
        label_model = LabelModel(cardinality=2, verbose=False)
        L = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
        label_model._set_constants(L)
        self.assertEqual(label_model.n, 4)
        self.assertEqual(label_model.m, 3)

    def test_class_balance(self):
        label_model = LabelModel(cardinality=2, verbose=False)
        # Test class balance
        Y_dev = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
        label_model._set_class_balance(class_balance=None, Y_dev=Y_dev)
        np.testing.assert_array_almost_equal(label_model.p, np.array([0.6, 0.4]))

    def test_generate_O(self):
        L = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1]])
        label_model = self._set_up_model(L)

        # O = (1/n) * L^TL = L^TL/4
        true_O = np.array(
            [
                [3 / 4, 0, 0, 3 / 4, 1 / 2, 1 / 4],
                [0, 1 / 4, 1 / 4, 0, 1 / 4, 0],
                [0, 1 / 4, 1 / 4, 0, 1 / 4, 0],
                [3 / 4, 0, 0, 3 / 4, 1 / 2, 1 / 4],
                [1 / 2, 1 / 4, 1 / 4, 1 / 2, 3 / 4, 0],
                [1 / 4, 0, 0, 1 / 4, 0, 1 / 4],
            ]
        )
        np.testing.assert_array_almost_equal(label_model.O.numpy(), true_O)

        label_model = self._set_up_model(L)
        label_model._generate_O(L + 1, higher_order=False)
        true_O = np.array(
            [
                [3 / 4, 0, 0, 3 / 4, 1 / 2, 1 / 4],
                [0, 1 / 4, 1 / 4, 0, 1 / 4, 0],
                [0, 1 / 4, 1 / 4, 0, 1 / 4, 0],
                [3 / 4, 0, 0, 3 / 4, 1 / 2, 1 / 4],
                [1 / 2, 1 / 4, 1 / 4, 1 / 2, 3 / 4, 0],
                [1 / 4, 0, 0, 1 / 4, 0, 1 / 4],
            ]
        )
        np.testing.assert_array_almost_equal(label_model.O.numpy(), true_O)

        # Higher order returns same matrix (num source = num cliques)
        # Need to test c_tree form
        label_model._generate_O(L + 1, higher_order=True)
        np.testing.assert_array_almost_equal(label_model.O.numpy(), true_O)

    def test_augmented_L_construction(self):
        # 5 LFs
        n = 3
        m = 5
        k = 2
        L = np.array([[0, 0, 0, 1, 0], [0, 1, 1, 0, -1], [0, 0, 0, 0, -1]])
        L_shift = L + 1
        lm = LabelModel(cardinality=k, verbose=False)
        lm._set_constants(L_shift)
        lm._create_tree()
        L_aug = lm._get_augmented_label_matrix(L_shift, higher_order=True)

        # Should have 10 columns:
        # - 5 * 2 = 10 for the sources
        self.assertEqual(L_aug.shape, (3, 10))

        # 13 total nonzero entries
        self.assertEqual(L_aug.sum(), 13)

        # Next, check the singleton entries
        for i in range(n):
            for j in range(m):
                if L_shift[i, j] > 0:
                    self.assertEqual(L_aug[i, j * k + L_shift[i, j] - 1], 1)

        # Finally, check the clique entries
        # Singleton clique 1
        self.assertEqual(len(lm.c_tree.node[1]["members"]), 1)
        j = lm.c_tree.node[1]["start_index"]
        self.assertEqual(L_aug[0, j], 1)

        # Singleton clique 2
        self.assertEqual(len(lm.c_tree.node[2]["members"]), 1)
        j = lm.c_tree.node[2]["start_index"]
        self.assertEqual(L_aug[0, j + 1], 0)

    def test_conditional_probs(self):
        L = np.array([[0, 1, 0], [0, 1, 0]])
        label_model = self._set_up_model(L, class_balance=[0.6, 0.4])
        probs = label_model._get_conditional_probs()
        self.assertLessEqual(probs.max(), 1.0)
        self.assertGreaterEqual(probs.min(), 0.0)

    def test_get_accuracy(self):
        L = np.array([[0, 1], [0, -1]])
        label_model = self._set_up_model(L)
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

        # accs[i] = sum(diag(probs[i*(k+1):(i+1)*(k+1)][1:,:]) * P)
        # accs[0] = 0.5*0.9 + 0.5*0.5 = 0.7
        # accs[1] = 0.5*0.9 + 0.5*0.75 = 0.825
        np.testing.assert_array_almost_equal(accs, np.array([0.7, 0.825]))

        # accs[i] = sum(diag(probs[i*(k+1):(i+1)*(k+1)][1:,:]) * P)
        # accs[0] = 0.5*0.5 + 0.5*0.5 = 0.5 since (P(\lambda_0 = 1 | Y = y) = 0.5)
        # accs[1] = 0.5*0.7 + 0.5*0.01 = 0.355 since (P(\lambda_1 = 1 | Y = 1) = 0.7)
        # default prec_init = 0.7, clamp lowest prob to 0.01
        label_model.mu = nn.Parameter(label_model.mu_init.clone())
        accs = label_model.get_accuracies(probs=None)
        np.testing.assert_array_almost_equal(accs, np.array([0.5, 0.355]))

    def test_build_mask(self):

        L = np.array([[0, 1, 0], [0, 1, 0]])
        label_model = self._set_up_model(L)

        # block diagonal with 0s for dependent LFs
        # without deps, k X k block of 0s down diagonal
        true_mask = np.array(
            [
                [0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
            ]
        )

        mask = label_model.mask.numpy()
        np.testing.assert_array_equal(mask, true_mask)

    def test_init_params(self):
        L = np.array([[0, 1, 0], [0, -1, 0]])
        label_model = self._set_up_model(L, class_balance=[0.6, 0.4])

        # mu_init = P(\lf=y|Y=y) = clamp(P(\lf=y) * prec_i / P(Y=y), (0,1))
        # mu_init[lf0, lf2 = 1 | Y = 1] = clamp(1.0 * 0.7 / 0.6) = 1.0 since P(lf = 1) = 1.0
        # mu_init[lf1 | Y = 2] = clamp(0.5 * 0.7 / 0.4) = 0.875 since P(lf = 2) = 0.5
        mu_init = label_model.mu_init.numpy()
        true_mu_init = np.array(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.875], [1.0, 0.0], [0.0, 0.0]]
        )
        np.testing.assert_array_equal(mu_init, true_mu_init)

        # mu_init = P(\lf=y|Y=y) = clamp(P(\lf=y) * prec_i / P(Y=y), (0,1))
        # mu_init[lf0, lf2 = 1 | Y = 1] = clamp(1.0 * 0.7 / 0.6) = 1.0 since P(lf = 1) = 1.0
        # mu_init[lf1 = 2 | Y = 2] = clamp(0.5 * 0.7 / 0.7) = 0.5 since P(lf = 2) = 0.5
        label_model._set_class_balance(class_balance=[0.3, 0.7], Y_dev=None)
        label_model._init_params()

        mu_init = label_model.mu_init.numpy()
        true_mu_init = np.array(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 0.0]]
        )
        np.testing.assert_array_equal(mu_init, true_mu_init)

    def test_predict_proba(self):
        L = np.array([[0, 1, 0], [0, 1, 0]])
        label_model = self._set_up_model(L)

        label_model.mu = nn.Parameter(label_model.mu_init.clone())
        probs = label_model.predict_proba(L)

        # with matching labels from 3 LFs, predicted probs clamped at (0.99,0.01)
        true_probs = np.array([[0.99, 0.01], [0.99, 0.01]])
        np.testing.assert_array_almost_equal(probs, true_probs)

    def test_predict(self):
        L = np.array([[0, 1, 0], [0, 1, 0]])
        label_model = self._set_up_model(L)

        label_model.mu = nn.Parameter(label_model.mu_init.clone())
        preds = label_model.predict(L)

        true_preds = np.array([0, 0])
        np.testing.assert_array_equal(preds, true_preds)

        preds, probs = label_model.predict(L, return_probs=True)
        true_probs = np.array([[0.99, 0.01], [0.99, 0.01]])
        np.testing.assert_array_almost_equal(probs, true_probs)

    def test_score(self):
        L = np.array([[1, 0, 1], [1, 0, 1]])
        label_model = self._set_up_model(L)
        label_model.mu = nn.Parameter(label_model.mu_init.clone())

        results = label_model.score(L, Y=np.array([0, 1]))
        results_expected = dict(accuracy=0.5)
        self.assertEqual(results, results_expected)

        results = label_model.score(L=L, Y=np.array([1, 0]), metrics=["accuracy", "f1"])
        results_expected = dict(accuracy=0.5, f1=2 / 3)
        self.assertEqual(results, results_expected)

    def test_loss(self):
        L = np.array([[0, -1, 0], [0, 1, -1]])
        label_model = self._set_up_model(L)
        label_model._get_augmented_label_matrix(L + 1, higher_order=True)

        label_model.mu = nn.Parameter(label_model.mu_init.clone() + 0.05)

        # l2_loss = l2*M*K*||mu - mu_init||_2 = 3*2*(0.05^2) = 0.03
        self.assertAlmostEqual(label_model._loss_l2(l2=1.0).item(), 0.03)
        self.assertAlmostEqual(label_model._loss_l2(l2=np.ones(6)).item(), 0.03)

        # mu_loss = ||O - \mu^T P \mu||_2 + ||\mu^T P - diag(O)||_2
        self.assertAlmostEqual(label_model._loss_mu().item(), 0.675, 3)

    def test_model_loss(self):
        L = np.array([[0, -1, 0], [0, 1, 0]])
        label_model = LabelModel(cardinality=2, verbose=False)

        label_model.train_model(L, n_epochs=1, lr=0.01, momentum=0.9)
        init_loss = label_model._loss_mu().item()

        label_model.train_model(L, n_epochs=10, lr=0.01, momentum=0.9)
        next_loss = label_model._loss_mu().item()

        self.assertLessEqual(next_loss, init_loss)

        with self.assertRaisesRegex(Exception, "Loss is NaN."):
            label_model.train_model(L, n_epochs=10, lr=1e8)

    def test_optimizer(self):
        L = np.array([[0, -1, 0], [0, 1, 0]])
        label_model = LabelModel(cardinality=2, verbose=False)
        label_model.train_model(L, n_epochs=1, optimizer="rmsprop")
        label_model.train_model(L, n_epochs=1, optimizer="adam")
        with self.assertRaisesRegex(ValueError, "Did not recognize optimizer"):
            label_model.train_model(L, n_epochs=1, optimizer="bad_opt")

    def test_lr_scheduler(self):
        L = np.array([[0, -1, 0], [0, 1, 0]])
        label_model = LabelModel(cardinality=2, verbose=False)
        label_model.train_model(L, n_epochs=1, lr_scheduler=None)
        label_model.train_model(L, n_epochs=1, lr_scheduler="exponential")
        with self.assertRaisesRegex(
            ValueError, "Did not recognize lr_scheduler option"
        ):
            label_model.train_model(L, n_epochs=1, lr_scheduler="bad_scheduler")

    def test_save_and_load(self):
        L = np.array([[0, -1, 0], [0, 1, 0]])
        label_model = LabelModel(cardinality=2, verbose=False)
        label_model.train_model(L, n_epochs=1, lr_scheduler=None)
        dir_path = tempfile.mkdtemp()
        save_path = dir_path + "label_model"
        label_model.save(save_path)
        label_model.load(save_path)
        shutil.rmtree(dir_path)


@pytest.mark.complex
class TestLabelModelAdvanced(unittest.TestCase):
    """Advanced (marked complex) tests for the LabelModel."""

    def setUp(self) -> None:
        """Set constants for the tests."""
        self.m = 10  # Number of LFs
        self.n = 10000  # Number of data points
        self.cardinality = 2  # Number of classes

    def test_label_model(self) -> None:
        """Test the LabelModel's estimate of P and Y."""
        np.random.seed(123)
        P, Y, L = generate_simple_label_matrix(self.n, self.m, self.cardinality)

        # Train LabelModel
        label_model = LabelModel(cardinality=self.cardinality, verbose=False)
        label_model.train_model(L, lr=0.01, l2=0.0, n_epochs=100)

        # Test estimated LF conditional probabilities
        P_lm = label_model._get_conditional_probs().reshape(
            (self.m, self.cardinality + 1, -1)
        )
        np.testing.assert_array_almost_equal(P, P_lm, decimal=2)

        # Test predicted labels
        Y_lm = label_model.predict_proba(L).argmax(axis=1)
        err = np.where(Y != Y_lm, 1, 0).sum() / self.n
        self.assertLess(err, 0.1)


if __name__ == "__main__":
    unittest.main()
