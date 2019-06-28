import unittest

import numpy as np

from snorkel.end_model.scorer import Scorer


class ScorerTest(unittest.TestCase):
    def test_scorer(self):
        golds = np.array([1, 0, 1, 0, 1])
        preds = np.array([1, 0, 1, 1, 0])
        probs = np.array([0.8, 0.6, 0.9, 0.7, 0.4])

        def sum(golds, preds, probs):
            return np.sum(preds)

        scorer = Scorer(metrics=["accuracy", "f1"], custom_metric_funcs={"sum": sum})

        results = scorer.score(golds, preds, probs)
        self.assertEqual(results["accuracy"], 0.6)
        self.assertEqual(results["f1"], 2 / 3)
        self.assertEqual(results["sum"], 3)
