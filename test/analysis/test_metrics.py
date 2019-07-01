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
        gold = [1, 1, 1, 2, 2]
        pred = [1, 1, 1, 2, 1]
        score = accuracy_score(gold, pred)
        self.assertAlmostEqual(score, 0.8)

    def test_metric_score(self):
        gold = [1, 1, 1, 2, 2]
        pred = [1, 1, 1, 2, 1]
        acc = accuracy_score(gold, pred)
        met = metric_score(gold, pred, prob=None, metric="accuracy")
        self.assertAlmostEqual(acc, met)

    def test_bad_inputs(self):
        gold = [1, 1, 1, 2, 2]
        pred1 = [1, 1, 1, 2, 0.5]
        pred2 = "1 1 1 2 2"
        pred3 = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
        self.assertRaises(ValueError, accuracy_score, gold, pred1)
        self.assertRaises(ValueError, accuracy_score, gold, pred2)
        self.assertRaises(ValueError, accuracy_score, gold, pred3)

    def test_array_conversion(self):
        gold = torch.Tensor([1, 1, 1, 2, 2])
        pred = np.array([1.0, 1.0, 1.0, 2.0, 1.0])
        score = accuracy_score(gold, pred)
        self.assertAlmostEqual(score, 0.8)

    def test_ignores(self):
        gold = [1, 1, 1, 2, 2]
        pred = [1, 0, 1, 2, 1]
        score = accuracy_score(gold, pred)
        self.assertAlmostEqual(score, 0.6)
        score = accuracy_score(gold, pred, ignore_in_pred=[0])
        self.assertAlmostEqual(score, 0.75)
        score = accuracy_score(gold, pred, ignore_in_gold=[1])
        self.assertAlmostEqual(score, 0.5)
        score = accuracy_score(gold, pred, ignore_in_gold=[2], ignore_in_pred=[0])
        self.assertAlmostEqual(score, 1.0)

    def test_coverage(self):
        gold = [1, 1, 1, 1, 2]
        pred = [0, 0, 1, 1, 1]
        score = coverage_score(gold, pred)
        self.assertAlmostEqual(score, 0.6)
        score = coverage_score(gold, pred, ignore_in_gold=[2])
        self.assertAlmostEqual(score, 0.5)

    def test_precision(self):
        gold = [1, 1, 1, 2, 2]
        pred = [0, 0, 1, 1, 2]
        score = precision_score(gold, pred)
        self.assertAlmostEqual(score, 0.5)
        score = precision_score(gold, pred, pos_label=2)
        self.assertAlmostEqual(score, 1.0)

    def test_recall(self):
        gold = [1, 1, 1, 1, 2]
        pred = [0, 2, 1, 1, 2]
        score = recall_score(gold, pred)
        self.assertAlmostEqual(score, 0.5)
        score = recall_score(gold, pred, pos_label=2)
        self.assertAlmostEqual(score, 1.0)

    def test_f1(self):
        gold = [1, 1, 1, 1, 2]
        pred = [0, 2, 1, 1, 2]
        score = f1_score(gold, pred)
        self.assertAlmostEqual(score, 0.666, places=2)
        score = f1_score(gold, pred, pos_label=2)
        self.assertAlmostEqual(score, 0.666, places=2)

    def test_fbeta(self):
        gold = [1, 1, 1, 1, 2]
        pred = [0, 2, 1, 1, 2]
        pre = precision_score(gold, pred)
        rec = recall_score(gold, pred)
        self.assertEqual(pre, fbeta_score(gold, pred, beta=0))
        self.assertAlmostEqual(rec, fbeta_score(gold, pred, beta=1000), places=4)


if __name__ == "__main__":
    unittest.main()
