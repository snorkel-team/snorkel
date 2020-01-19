import shutil
import tempfile
import unittest
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from snorkel.labeling.model import LabelModel
from snorkel.labeling.model.label_model import TrainConfig
from snorkel.synthetic.synthetic_data import generate_simple_label_matrix


class LabelModelTest(unittest.TestCase):
    def _set_up_model(self, L: np.ndarray, class_balance: List[float] = [0.5, 0.5]):
        label_model = LabelModel(cardinality=2, verbose=False)
        label_model.train_config = TrainConfig()  # type: ignore
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
        L = np.array([[-1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, 1, -1]])
        label_model._set_constants(L)
        self.assertEqual(label_model.n, 4)
        self.assertEqual(label_model.m, 3)

        L = np.array([[-1, 0, 1], [-1, 0, 2], [0, -1, 2], [-1, 0, -1]])
        with self.assertRaisesRegex(ValueError, "L_train has cardinality"):
            label_model.fit(L, n_epochs=1)

        L = np.array([[0, 1], [1, 1], [0, 1]])
        with self.assertRaisesRegex(ValueError, "L_train should have at least 3"):
            label_model.fit(L, n_epochs=1)

    def test_mv_default(self):
        # less than 2 LFs have overlaps
        label_model = LabelModel(cardinality=2, verbose=False)
        L = np.array([[-1, -1, 1], [-1, 1, -1], [0, -1, -1]])
        label_model.fit(L, n_epochs=100)
        np.testing.assert_array_almost_equal(
            label_model.predict(L), np.array([1, 1, 0])
        )

        # less than 2 LFs have conflicts
        L = np.array([[-1, -1, 1], [-1, 1, 1], [1, 1, 1]])
        label_model.fit(L, n_epochs=100)
        np.testing.assert_array_almost_equal(
            label_model.predict(L), np.array([1, 1, 1])
        )

    def test_class_balance(self):
        label_model = LabelModel(cardinality=2, verbose=False)
        # Test class balance
        Y_dev = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
        label_model._set_class_balance(class_balance=None, Y_dev=Y_dev)
        np.testing.assert_array_almost_equal(label_model.p, np.array([0.6, 0.4]))

        class_balance = np.array([0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "Class balance prior is 0"):
            label_model._set_class_balance(class_balance=class_balance, Y_dev=Y_dev)

        class_balance = np.array([0.0])
        with self.assertRaisesRegex(ValueError, "class_balance has 1 entries."):
            label_model._set_class_balance(class_balance=class_balance, Y_dev=Y_dev)

        Y_dev_one_class = np.array([0, 0, 0])
        with self.assertRaisesRegex(
            ValueError, "Does not match LabelModel cardinality"
        ):
            label_model._set_class_balance(class_balance=None, Y_dev=Y_dev_one_class)

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
        np.testing.assert_array_almost_equal(
            label_model.O.cpu().detach().numpy(), true_O
        )

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
        np.testing.assert_array_almost_equal(
            label_model.O.cpu().detach().numpy(), true_O
        )

        # Higher order returns same matrix (num source = num cliques)
        # Need to test c_tree form
        label_model._generate_O(L + 1, higher_order=True)
        np.testing.assert_array_almost_equal(
            label_model.O.cpu().detach().numpy(), true_O
        )

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
        cprobs = label_model.get_conditional_probs()
        self.assertLessEqual(cprobs.max(), 1.0)
        self.assertGreaterEqual(cprobs.min(), 0.0)

    def test_get_weight(self):
        # set up L matrix
        true_accs = [0.95, 0.6, 0.7, 0.55, 0.8]
        coverage = [1.0, 0.8, 1.0, 1.0, 1.0]
        L = -1 * np.ones((1000, len(true_accs)))
        Y = np.zeros(1000)

        for i in range(1000):
            Y[i] = 1 if np.random.rand() <= 0.5 else 0
            for j in range(5):
                if np.random.rand() <= coverage[j]:
                    L[i, j] = (
                        Y[i] if np.random.rand() <= true_accs[j] else np.abs(Y[i] - 1)
                    )

        label_model = LabelModel(cardinality=2)
        label_model.fit(L, n_epochs=1000, seed=123)

        accs = label_model.get_weights()
        for i in range(len(accs)):
            true_acc = true_accs[i]
            self.assertAlmostEqual(accs[i], true_acc, delta=0.1)

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

        label_model.mu = nn.Parameter(label_model.mu_init.clone().clamp(0.01, 0.99))
        probs = label_model.predict_proba(L)

        # with matching labels from 3 LFs, predicted probs clamped at (0.99,0.01)
        true_probs = np.array([[0.99, 0.01], [0.99, 0.01]])
        np.testing.assert_array_almost_equal(probs, true_probs)

    def test_predict(self):
        # 3 LFs that always disagree/abstain leads to all abstains
        L = np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
        label_model = LabelModel(cardinality=2, verbose=False)
        label_model.fit(L, n_epochs=100)
        np.testing.assert_array_almost_equal(
            label_model.predict(L), np.array([-1, -1, -1])
        )

        L = np.array([[0, 1, 0], [0, 1, 0]])
        label_model = self._set_up_model(L)

        label_model.mu = nn.Parameter(label_model.mu_init.clone().clamp(0.01, 0.99))
        preds = label_model.predict(L)

        true_preds = np.array([0, 0])
        np.testing.assert_array_equal(preds, true_preds)

        preds, probs = label_model.predict(L, return_probs=True)
        true_probs = np.array([[0.99, 0.01], [0.99, 0.01]])
        np.testing.assert_array_almost_equal(probs, true_probs)

    def test_score(self):
        L = np.array([[1, 1, 0], [-1, -1, -1], [1, 0, 1]])
        Y = np.array([1, 0, 1])
        label_model = LabelModel(cardinality=2, verbose=False)
        label_model.fit(L, n_epochs=100)
        results = label_model.score(L, Y, metrics=["accuracy", "coverage"])
        np.testing.assert_array_almost_equal(
            label_model.predict(L), np.array([1, -1, 1])
        )

        results_expected = dict(accuracy=1.0, coverage=2 / 3)
        self.assertEqual(results, results_expected)

        L = np.array([[1, 0, 1], [1, 0, 1]])
        label_model = self._set_up_model(L)
        label_model.mu = nn.Parameter(label_model.mu_init.clone().clamp(0.01, 0.99))

        results = label_model.score(L, Y=np.array([0, 1]))
        results_expected = dict(accuracy=0.5)
        self.assertEqual(results, results_expected)

        results = label_model.score(L=L, Y=np.array([1, 0]), metrics=["accuracy", "f1"])
        results_expected = dict(accuracy=0.5, f1=2 / 3)
        self.assertEqual(results, results_expected)

    def test_loss(self):
        L = np.array([[0, -1, 0], [0, 1, -1]])
        label_model = LabelModel(cardinality=2, verbose=False)
        label_model.fit(L, n_epochs=1)
        label_model.mu = nn.Parameter(label_model.mu_init.clone() + 0.05)

        # l2_loss = l2*M*K*||mu - mu_init||_2 = 3*2*(0.05^2) = 0.03
        self.assertAlmostEqual(label_model._loss_l2(l2=1.0).item(), 0.03)
        self.assertAlmostEqual(label_model._loss_l2(l2=np.ones(6)).item(), 0.03)

        # mu_loss = ||O - \mu^T P \mu||_2 + ||\mu^T P - diag(O)||_2
        self.assertAlmostEqual(label_model._loss_mu().item(), 0.675, 3)

    def test_model_loss(self):
        L = np.array([[0, -1, 0], [0, 1, 0]])
        label_model = LabelModel(cardinality=2, verbose=False)

        label_model.fit(L, n_epochs=1)
        init_loss = label_model._loss_mu().item()

        label_model.fit(L, n_epochs=10)
        next_loss = label_model._loss_mu().item()

        self.assertLessEqual(next_loss, init_loss)

        with self.assertRaisesRegex(Exception, "Loss is NaN."):
            label_model.fit(L, n_epochs=10, lr=1e8)

    def test_optimizer(self):
        L = np.array([[0, -1, 0], [0, 1, 0]])
        label_model = LabelModel(cardinality=2, verbose=False)
        label_model.fit(L, n_epochs=1, optimizer="sgd")
        label_model.fit(L, n_epochs=1, optimizer="adam")
        label_model.fit(L, n_epochs=1, optimizer="adamax")
        with self.assertRaisesRegex(ValueError, "Unrecognized optimizer option"):
            label_model.fit(L, n_epochs=1, optimizer="bad_opt")

    def test_lr_scheduler(self):
        L = np.array([[0, -1, 0], [0, 1, 0]])
        label_model = LabelModel(cardinality=2, verbose=False)
        label_model.fit(L, n_epochs=1)
        label_model.fit(L, n_epochs=1, lr_scheduler="constant")
        label_model.fit(L, n_epochs=1, lr_scheduler="linear")
        label_model.fit(L, n_epochs=1, lr_scheduler="exponential")
        label_model.fit(L, n_epochs=1, lr_scheduler="step")
        with self.assertRaisesRegex(ValueError, "Unrecognized lr scheduler option"):
            label_model.fit(L, n_epochs=1, lr_scheduler="bad_scheduler")

    def test_save_and_load(self):
        L = np.array([[0, -1, 0], [0, 1, 1]])
        label_model = LabelModel(cardinality=2, verbose=False)
        label_model.fit(L, n_epochs=1)
        original_preds = label_model.predict(L)

        dir_path = tempfile.mkdtemp()
        save_path = dir_path + "label_model.pkl"
        label_model.save(save_path)

        label_model_new = LabelModel(cardinality=2, verbose=False)
        label_model_new.load(save_path)
        loaded_preds = label_model_new.predict(L)
        shutil.rmtree(dir_path)

        np.testing.assert_array_equal(loaded_preds, original_preds)

    def test_optimizer_init(self):
        L = np.array([[0, -1, 0], [0, 1, 0]])
        label_model = LabelModel()

        label_model.fit(L, optimizer="sgd", n_epochs=1)
        self.assertIsInstance(label_model.optimizer, optim.SGD)

        label_model.fit(L, optimizer="adam", n_epochs=1)
        self.assertIsInstance(label_model.optimizer, optim.Adam)

        label_model.fit(L, optimizer="adamax", n_epochs=1)
        self.assertIsInstance(label_model.optimizer, optim.Adamax)

        with self.assertRaisesRegex(ValueError, "Unrecognized optimizer"):
            label_model.fit(L, optimizer="bad_optimizer", n_epochs=1)

    def test_scheduler_init(self):
        L = np.array([[0, -1, 0], [0, 1, 0]])
        label_model = LabelModel()

        label_model.fit(L, lr_scheduler="constant", n_epochs=1)
        self.assertIsNone(label_model.lr_scheduler)

        label_model.fit(L, lr_scheduler="linear", n_epochs=1)
        self.assertIsInstance(label_model.lr_scheduler, optim.lr_scheduler.LambdaLR)

        label_model.fit(L, lr_scheduler="exponential", n_epochs=1)
        self.assertIsInstance(
            label_model.lr_scheduler, optim.lr_scheduler.ExponentialLR
        )

        label_model.fit(L, lr_scheduler="step", n_epochs=1)
        self.assertIsInstance(label_model.lr_scheduler, optim.lr_scheduler.StepLR)

    def test_warmup(self):
        L = np.array([[0, -1, 0], [0, 1, 0]])
        label_model = LabelModel()

        lr_scheduler_config = {"warmup_steps": 3, "warmup_unit": "epochs"}
        label_model.fit(L, lr_scheduler_config=lr_scheduler_config, n_epochs=5)
        self.assertEqual(label_model.warmup_steps, 3)

        lr_scheduler_config = {"warmup_percentage": 3 / 5}
        label_model.fit(L, lr_scheduler_config=lr_scheduler_config, n_epochs=5)
        self.assertEqual(label_model.warmup_steps, 3)

        with self.assertRaisesRegex(ValueError, "LabelModel does not support"):
            lr_scheduler_config = {"warmup_steps": 1, "warmup_unit": "batches"}
            label_model.fit(L, lr_scheduler_config=lr_scheduler_config)

    def test_set_mu_eps(self):
        mu_eps = 0.0123

        # Construct a label matrix such that P(\lambda_1 = 0 | Y) = 0.0, so it will hit
        # the mu_eps floor
        L = np.array([[1, 1, 1], [1, 1, 1]])
        label_model = LabelModel(verbose=False)
        label_model.fit(L, mu_eps=mu_eps)
        self.assertAlmostEqual(label_model.get_conditional_probs()[0, 1, 0], mu_eps)

    def test_symmetry_breaking(self):
        mu = np.array(
            [
                # LF 0
                [0.75, 0.25],
                [0.25, 0.75],
                # LF 1
                [0.25, 0.75],
                [0.15, 0.25],
                # LF 2
                [0.75, 0.25],
                [0.25, 0.75],
            ]
        )
        mu = mu[:, [1, 0]]

        # First test: Two "good" LFs
        label_model = LabelModel(verbose=False)
        label_model._set_class_balance(None, None)
        label_model.m = 3
        label_model.mu = nn.Parameter(torch.from_numpy(mu))
        label_model._break_col_permutation_symmetry()
        self.assertEqual(label_model.mu.data[0, 0], 0.75)

        # Test with non-uniform class balance
        # It should not consider the "correct" permutation as does not commute now
        label_model = LabelModel(verbose=False)
        label_model._set_class_balance([0.9, 0.1], None)
        label_model.m = 3
        label_model.mu = nn.Parameter(torch.from_numpy(mu))
        label_model._break_col_permutation_symmetry()
        self.assertEqual(label_model.mu.data[0, 0], 0.25)

    def test_symmetry_breaking_multiclass(self):
        mu = np.array(
            [
                # LF 0
                [0.75, 0.15, 0.1],
                [0.20, 0.75, 0.3],
                [0.05, 0.10, 0.6],
                # LF 1
                [0.25, 0.55, 0.3],
                [0.15, 0.45, 0.4],
                [0.20, 0.00, 0.3],
                # LF 2
                [0.5, 0.15, 0.2],
                [0.3, 0.65, 0.2],
                [0.2, 0.20, 0.6],
            ]
        )
        mu = mu[:, [1, 2, 0]]

        # First test: Two "good" LFs
        label_model = LabelModel(cardinality=3, verbose=False)
        label_model._set_class_balance(None, None)
        label_model.m = 3
        label_model.mu = nn.Parameter(torch.from_numpy(mu))
        label_model._break_col_permutation_symmetry()
        self.assertEqual(label_model.mu.data[0, 0], 0.75)
        self.assertEqual(label_model.mu.data[1, 1], 0.75)

        # Test with non-uniform class balance
        # It should not consider the "correct" permutation as it does not commute
        label_model = LabelModel(cardinality=3, verbose=False)
        label_model._set_class_balance([0.7, 0.2, 0.1], None)
        label_model.m = 3
        label_model.mu = nn.Parameter(torch.from_numpy(mu))
        label_model._break_col_permutation_symmetry()
        self.assertEqual(label_model.mu.data[0, 0], 0.15)
        self.assertEqual(label_model.mu.data[1, 1], 0.3)


@pytest.mark.complex
class TestLabelModelAdvanced(unittest.TestCase):
    """Advanced (marked complex) tests for the LabelModel."""

    def setUp(self) -> None:
        """Set constants for the tests."""
        self.m = 10  # Number of LFs
        self.n = 10000  # Number of data points
        self.cardinality = 2  # Number of classes

    def test_label_model_basic(self) -> None:
        """Test the LabelModel's estimate of P and Y on a simple synthetic dataset."""
        np.random.seed(123)
        P, Y, L = generate_simple_label_matrix(self.n, self.m, self.cardinality)

        # Train LabelModel
        label_model = LabelModel(cardinality=self.cardinality, verbose=False)
        label_model.fit(L, n_epochs=200, lr=0.01, seed=123)

        # Test estimated LF conditional probabilities
        P_lm = label_model.get_conditional_probs()
        np.testing.assert_array_almost_equal(P, P_lm, decimal=2)

        # Test predicted labels
        score = label_model.score(L, Y)
        self.assertGreaterEqual(score["accuracy"], 0.9)

    def test_label_model_sparse(self) -> None:
        """Test the LabelModel's estimate of P and Y on a sparse synthetic dataset.

        This tests the common setting where LFs abstain most of the time, which can
        cause issues for example if parameter clamping set too high (e.g. see Issue
        #1422).
        """
        np.random.seed(123)
        P, Y, L = generate_simple_label_matrix(
            self.n, self.m, self.cardinality, abstain_multiplier=1000.0
        )

        # Train LabelModel
        label_model = LabelModel(cardinality=self.cardinality, verbose=False)
        label_model.fit(L, n_epochs=1000, lr=0.01, seed=123)

        # Test estimated LF conditional probabilities
        P_lm = label_model.get_conditional_probs()
        np.testing.assert_array_almost_equal(P, P_lm, decimal=2)

        # Test predicted labels *only on non-abstained data points*
        Y_pred = label_model.predict(L, tie_break_policy="abstain")
        (idx,) = np.where(Y_pred != -1)
        acc = np.where(Y_pred[idx] == Y[idx], 1, 0).sum() / len(idx)
        self.assertGreaterEqual(acc, 0.65)

        # Make sure that we don't output abstain when an LF votes, per issue #1422
        self.assertEqual(len(idx), np.where((L + 1).sum(axis=1) != 0, 1, 0).sum())


if __name__ == "__main__":
    unittest.main()
