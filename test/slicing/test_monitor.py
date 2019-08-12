import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from snorkel.analysis.scorer import Scorer
from snorkel.slicing import SFApplier, slicing_function
from snorkel.slicing.monitor import SliceScorer, slice_dataframe
from snorkel.utils import preds_to_probs

DATA = [5, 10, 19, 22, 25]


@slicing_function()
def sf(x):
    return x.num < 20


class PandasSlicerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame(dict(num=DATA))

    def test_slice(self):
        self.assertEqual(len(self.df), 5)

        # Should return a subset
        sliced_df = slice_dataframe(self.df, sf)
        self.assertEqual(len(sliced_df), 3)


class SliceScorerTest(unittest.TestCase):
    def test_slice(self):
        # We expect 3/5 correct -> 0.6 accuracy
        golds = np.array([0, 1, 0, 1, 0])
        preds = np.array([0, 0, 0, 0, 0])
        probs = preds_to_probs(preds, 2)

        scorer = Scorer(metrics=["accuracy"])
        metrics = scorer.score(golds, preds, probs)
        self.assertEqual(metrics["accuracy"], 0.6)

        # In the slice, we expect the last 2 elements to masked
        # We expect 2/3 correct -> 0.666 accuracy
        data = [SimpleNamespace(num=x) for x in DATA]
        S_matrix = SFApplier([sf]).apply(data)
        slice_names = [sf.name]
        scorer = SliceScorer(scorer, slice_names)
        metrics = scorer.score(S=S_matrix, golds=golds, preds=preds, probs=probs)
        self.assertEqual(metrics["overall"]["accuracy"], 0.6)
        self.assertEqual(metrics["sf"]["accuracy"], 2.0 / 3.0)

        metrics_df = scorer.score(
            S=S_matrix, golds=golds, preds=preds, probs=probs, as_dataframe=True
        )
        self.assertTrue(isinstance(metrics_df, pd.DataFrame))
        self.assertEqual(metrics_df["accuracy"]["overall"], 0.6)
        self.assertEqual(metrics_df["accuracy"]["sf"], 2.0 / 3.0)
