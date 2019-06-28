import unittest

import numpy as np
import torch

from snorkel.analysis.metrics import (
    accuracy_score,
    coverage_score,
    f1_score,
    fbeta_score,
    metric_score,
    precision_score,
    recall_score,
)


class MetricsTest(unittest.TestCase):
    def test_accuracy_basic(self):
        golds = [1, 1, 1, 2, 2]
        preds = [1, 1, 1, 2, 1]
        score = accuracy_score(golds, preds)
        self.assertAlmostEqual(score, 0.8)

    def test_metric_score(self):
        golds = [1, 1, 1, 2, 2]
        preds = [1, 1, 1, 2, 1]
        acc = accuracy_score(golds, preds)
        met = metric_score(golds, preds, probs=None, metric="accuracy")
        self.assertAlmostEqual(acc, met)

    def test_bad_inputs(self):
        golds = [1, 1, 1, 2, 2]
        pred1 = [1, 1, 1, 2, 0.5]
        pred2 = "1 1 1 2 2"
        pred3 = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
        self.assertRaises(ValueError, accuracy_score, golds, pred1)
        self.assertRaises(ValueError, accuracy_score, golds, pred2)
        self.assertRaises(ValueError, accuracy_score, golds, pred3)

    def test_array_conversion(self):
        golds = torch.Tensor([1, 1, 1, 2, 2])
        preds = np.array([1.0, 1.0, 1.0, 2.0, 1.0])
        score = accuracy_score(golds, preds)
        self.assertAlmostEqual(score, 0.8)

    def test_ignores(self):
        golds = [1, 1, 1, 2, 2]
        preds = [1, 0, 1, 2, 1]
        score = accuracy_score(golds, preds)
        self.assertAlmostEqual(score, 0.6)
        score = accuracy_score(golds, preds, ignore_in_pred=[0])
        self.assertAlmostEqual(score, 0.75)
        score = accuracy_score(golds, preds, ignore_in_gold=[1])
        self.assertAlmostEqual(score, 0.5)
        score = accuracy_score(golds, preds, ignore_in_gold=[2], ignore_in_pred=[0])
        self.assertAlmostEqual(score, 1.0)

    def test_coverage(self):
        golds = [1, 1, 1, 1, 2]
        preds = [0, 0, 1, 1, 1]
        score = coverage_score(golds, preds)
        self.assertAlmostEqual(score, 0.6)
        score = coverage_score(golds, preds, ignore_in_gold=[2])
        self.assertAlmostEqual(score, 0.5)

    def test_precision(self):
        golds = [1, 1, 1, 2, 2]
        preds = [0, 0, 1, 1, 2]
        score = precision_score(golds, preds)
        self.assertAlmostEqual(score, 0.5)
        score = precision_score(golds, preds, pos_label=2)
        self.assertAlmostEqual(score, 1.0)

    def test_recall(self):
        golds = [1, 1, 1, 1, 2]
        preds = [0, 2, 1, 1, 2]
        score = recall_score(golds, preds)
        self.assertAlmostEqual(score, 0.5)
        score = recall_score(golds, preds, pos_label=2)
        self.assertAlmostEqual(score, 1.0)

    def test_f1(self):
        golds = [1, 1, 1, 1, 2]
        preds = [0, 2, 1, 1, 2]
        score = f1_score(golds, preds)
        self.assertAlmostEqual(score, 0.666, places=2)
        score = f1_score(golds, preds, pos_label=2)
        self.assertAlmostEqual(score, 0.666, places=2)

    def test_fbeta(self):
        golds = [1, 1, 1, 1, 2]
        preds = [0, 2, 1, 1, 2]
        pre = precision_score(golds, preds)
        rec = recall_score(golds, preds)
        self.assertEqual(pre, fbeta_score(golds, preds, beta=0))
        self.assertAlmostEqual(rec, fbeta_score(golds, preds, beta=1000), places=4)


if __name__ == "__main__":
    unittest.main()
