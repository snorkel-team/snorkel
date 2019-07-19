import unittest

import numpy as np

from snorkel.analysis.metrics import metric_score


class MetricsTest(unittest.TestCase):
    def test_accuracy_basic(self):
        golds = np.array([1, 1, 1, 2, 2])
        preds = np.array([1, 1, 1, 2, 1])
        score = metric_score(golds, preds, probs=None, metric="accuracy")
        self.assertAlmostEqual(score, 0.8)

    def test_bad_inputs(self):
        golds = np.array([1, 1, 1, 2, 2])
        pred1 = np.array([1, 1, 1, 2, 0.5])
        pred2 = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
        with self.assertRaisesRegex(
            ValueError, "Input contains at least one non-integer"
        ):
            metric_score(golds, pred1, probs=None, metric="accuracy")

        with self.assertRaisesRegex(ValueError, "Input could not be converted"):
            metric_score(golds, pred2, probs=None, metric="accuracy")

        with self.assertRaisesRegex(ValueError, "The metric you provided"):
            metric_score(golds, pred2, probs=None, metric="bad_metric")

        with self.assertRaisesRegex(
            ValueError, "filter_dict must only include keys in"
        ):
            metric_score(
                golds,
                golds,
                probs=None,
                metric="accuracy",
                filter_dict={"bad_map": [0]},
            )

    def test_ignores(self):
        golds = np.array([1, 1, 1, 2, 2])
        preds = np.array([1, 0, 1, 2, 1])
        score = metric_score(golds, preds, probs=None, metric="accuracy")
        self.assertAlmostEqual(score, 0.6)
        score = metric_score(
            golds, preds, probs=None, metric="accuracy", filter_dict={"preds": [0]}
        )
        self.assertAlmostEqual(score, 0.75)
        score = metric_score(
            golds, preds, probs=None, metric="accuracy", filter_dict={"golds": [1]}
        )
        self.assertAlmostEqual(score, 0.5)
        score = metric_score(
            golds,
            preds,
            probs=None,
            metric="accuracy",
            filter_dict={"golds": [2], "preds": [0]},
        )
        self.assertAlmostEqual(score, 1.0)

    def test_coverage(self):
        golds = np.array([1, 1, 1, 1, 2])
        preds = np.array([0, 0, 1, 1, 1])
        score = metric_score(golds, preds, probs=None, metric="coverage")
        self.assertAlmostEqual(score, 0.6)
        score = metric_score(
            golds, preds, probs=None, filter_dict={"golds": [2]}, metric="coverage"
        )
        self.assertAlmostEqual(score, 0.5)

    def test_precision(self):
        golds = np.array([1, 1, 1, 2, 2])
        preds = np.array([2, 2, 1, 1, 2])
        score = metric_score(golds, preds, probs=None, metric="precision")
        self.assertAlmostEqual(score, 0.5)
        score = metric_score(golds, preds, probs=None, metric="precision", pos_label=2)
        self.assertAlmostEqual(score, 0.333, places=2)

    def test_recall(self):
        golds = np.array([1, 1, 1, 1, 2])
        preds = np.array([2, 2, 1, 1, 2])
        score = metric_score(golds, preds, probs=None, metric="recall")
        self.assertAlmostEqual(score, 0.5)
        score = metric_score(golds, preds, probs=None, metric="recall", pos_label=2)
        self.assertAlmostEqual(score, 1.0)

    def test_f1(self):
        golds = np.array([1, 1, 1, 1, 2])
        preds = np.array([2, 2, 1, 1, 2])
        score = metric_score(golds, preds, probs=None, metric="f1")
        self.assertAlmostEqual(score, 0.666, places=2)
        score = metric_score(golds, preds, probs=None, pos_label=2, metric="f1")
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_fbeta(self):
        golds = np.array([1, 1, 1, 1, 2])
        preds = np.array([2, 2, 1, 1, 2])
        pre = metric_score(golds, preds, probs=None, metric="precision")
        rec = metric_score(golds, preds, probs=None, metric="recall")
        self.assertAlmostEqual(
            pre,
            metric_score(golds, preds, probs=None, metric="fbeta", beta=1e-6),
            places=2,
        )
        self.assertAlmostEqual(
            rec,
            metric_score(golds, preds, probs=None, metric="fbeta", beta=1e6),
            places=2,
        )

    def test_matthews(self):
        golds = np.array([1, 1, 1, 1, 2])
        preds = np.array([2, 1, 1, 1, 1])
        mcc = metric_score(golds, preds, probs=None, metric="matthews_corrcoef")
        self.assertAlmostEqual(mcc, -0.25)

        golds = np.array([1, 1, 1, 1, 2])
        preds = np.array([1, 1, 1, 1, 2])
        mcc = metric_score(golds, preds, probs=None, metric="matthews_corrcoef")
        self.assertAlmostEqual(mcc, 1.0)

    def test_roc_auc(self):
        golds = np.array([1, 1, 1, 1, 2])
        probs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        probs_nonbinary = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.7, 0.0, 0.3],
                [0.8, 0.0, 0.2],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        roc_auc = metric_score(golds, preds=None, probs=probs, metric="roc_auc")
        self.assertAlmostEqual(roc_auc, 0.0)
        probs = np.fliplr(probs)
        roc_auc = metric_score(golds, preds=None, probs=probs, metric="roc_auc")
        self.assertAlmostEqual(roc_auc, 1.0)

        with self.assertRaisesRegex(
            ValueError, "Metric roc_auc is currently only defined for binary"
        ):
            metric_score(golds, preds=None, probs=probs_nonbinary, metric="roc_auc")

    def test_missing_preds(self):
        golds = np.array([1, 1, 2, 2])
        with self.assertRaisesRegex(ValueError, "requires access to"):
            metric_score(golds=golds, metric="accuracy")

    def test_probs_to_preds_conversion(self):
        golds = np.array([1, 1, 2, 2])
        probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        self.assertEqual(metric_score(golds=golds, probs=probs, metric="accuracy"), 0.5)


if __name__ == "__main__":
    unittest.main()
