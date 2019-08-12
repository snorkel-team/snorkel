import unittest

import numpy as np
import torch
import torch.nn.functional as F

from snorkel.classification import cross_entropy_with_probs
from snorkel.utils import preds_to_probs


class SoftCrossEntropyTest(unittest.TestCase):
    def test_sce_equals_ce(self):
        # Does soft ce loss match classic ce loss when labels are one-hot?
        Y_golds = torch.LongTensor([0, 1, 2])
        Y_golds_probs = torch.Tensor(preds_to_probs(Y_golds.numpy(), num_classes=4))

        Y_probs = torch.rand_like(Y_golds_probs)
        Y_probs = Y_probs / Y_probs.sum(dim=1).reshape(-1, 1)

        ce_loss = F.cross_entropy(Y_probs, Y_golds, reduction="none")
        ces_loss = cross_entropy_with_probs(Y_probs, Y_golds_probs, reduction="none")
        np.testing.assert_equal(ce_loss.numpy(), ces_loss.numpy())

        ce_loss = F.cross_entropy(Y_probs, Y_golds, reduction="sum")
        ces_loss = cross_entropy_with_probs(Y_probs, Y_golds_probs, reduction="sum")
        np.testing.assert_equal(ce_loss.numpy(), ces_loss.numpy())

        ce_loss = F.cross_entropy(Y_probs, Y_golds, reduction="mean")
        ces_loss = cross_entropy_with_probs(Y_probs, Y_golds_probs, reduction="mean")
        np.testing.assert_equal(ce_loss.numpy(), ces_loss.numpy())

    def test_perfect_predictions(self):
        # Does soft ce loss achieve approx. 0 loss with perfect predictions?
        Y_golds = torch.LongTensor([0, 1, 2])
        Y_golds_probs = torch.Tensor(preds_to_probs(Y_golds.numpy(), num_classes=4))

        Y_probs = Y_golds_probs.clone()
        Y_probs[Y_probs == 1] = 100
        Y_probs[Y_probs == 0] = -100

        ces_loss = cross_entropy_with_probs(Y_probs, Y_golds_probs)
        np.testing.assert_equal(ces_loss.numpy(), 0)

    def test_lower_loss(self):
        # Is loss lower when it should be?
        Y_golds_probs = torch.tensor([[0.1, 0.9], [0.5, 0.5]])
        Y_probs1 = torch.tensor([[0.1, 0.3], [1.0, 0.0]])
        Y_probs2 = torch.tensor([[0.1, 0.2], [1.0, 0.0]])
        ces_loss1 = cross_entropy_with_probs(Y_probs1, Y_golds_probs)
        ces_loss2 = cross_entropy_with_probs(Y_probs2, Y_golds_probs)
        self.assertLess(ces_loss1, ces_loss2)

    def test_equal_loss(self):
        # Is loss equal when it should be?
        Y_golds_probs = torch.tensor([[0.1, 0.9], [0.5, 0.5]])
        Y_probs1 = torch.tensor([[0.1, 0.3], [1.0, 0.0]])
        Y_probs2 = torch.tensor([[0.1, 0.3], [0.0, 1.0]])
        ces_loss1 = cross_entropy_with_probs(Y_probs1, Y_golds_probs)
        ces_loss2 = cross_entropy_with_probs(Y_probs2, Y_golds_probs)
        self.assertEqual(ces_loss1, ces_loss2)

    def test_invalid_reduction(self):
        Y_golds = torch.LongTensor([0, 1, 2])
        Y_golds_probs = torch.Tensor(preds_to_probs(Y_golds.numpy(), num_classes=4))

        Y_probs = torch.rand_like(Y_golds_probs)
        Y_probs = Y_probs / Y_probs.sum(dim=1).reshape(-1, 1)

        with self.assertRaisesRegex(ValueError, "Keyword 'reduction' must be"):
            cross_entropy_with_probs(Y_probs, Y_golds_probs, reduction="bad")

    def test_loss_weights(self):
        FACTOR = 10

        # Do class weights work as expected?
        Y_golds = torch.LongTensor([0, 0, 1])
        Y_golds_probs = torch.Tensor(preds_to_probs(Y_golds.numpy(), num_classes=3))
        # Predict [1, 1, 1]
        Y_probs = torch.tensor(
            [[-100.0, 100.0, -100.0], [-100.0, 100.0, -100.0], [-100.0, 100.0, -100.0]]
        )

        ces_loss0 = cross_entropy_with_probs(Y_probs, Y_golds_probs).numpy()
        weight1 = torch.FloatTensor([1, 1, 1])
        ces_loss1 = cross_entropy_with_probs(
            Y_probs, Y_golds_probs, weight=weight1
        ).numpy()
        # Do weights of 1 match no weights at all?
        self.assertEqual(ces_loss0, ces_loss1)

        weight2 = torch.FloatTensor([1, 2, 1])
        ces_loss2 = cross_entropy_with_probs(
            Y_probs, Y_golds_probs, weight=weight2
        ).numpy()
        weight3 = weight2 * FACTOR
        ces_loss3 = cross_entropy_with_probs(
            Y_probs, Y_golds_probs, weight=weight3
        ).numpy()
        # If weights are X times larger, is loss X times larger?
        self.assertAlmostEqual(ces_loss2 * FACTOR, ces_loss3, places=3)

        # Note that PyTorch's cross-entropy loss has the unusual behavior that weights
        # behave differently when losses are averaged inside vs. outside the function.
        # See https://github.com/pytorch/pytorch/issues/8062 for details.
        ce_loss3 = (
            F.cross_entropy(Y_probs, Y_golds, weight=weight3, reduction="none")
            .mean()
            .numpy()
        )
        # Do hard and soft ce loss still match when we use class weights?
        self.assertAlmostEqual(ce_loss3, ces_loss3, places=3)


if __name__ == "__main__":
    unittest.main()
