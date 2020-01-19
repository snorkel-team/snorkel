import random
import unittest

import numpy as np
import pandas as pd
import pytest
import torch

from snorkel.labeling import LabelingFunction, PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel
from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint


def create_data(n: int) -> pd.DataFrame:
    """Create random pairs x1, x2 in [-1., 1.] with label x1 > x2 + 0.25."""
    X = np.random.random((n, 2)) * 2 - 1
    Y = (X[:, 0] > X[:, 1] + 0.25).astype(int)

    df = pd.DataFrame(
        {"x0": np.random.randint(0, 1000, n), "x1": X[:, 0], "x2": X[:, 1], "y": Y}
    )
    return df


def get_positive_labeling_function(divisor: int) -> LabelingFunction:
    """Get LabelingFunction that abstains unless x0 is divisible by divisor."""

    def f(x):
        return 1 if x.x0 % divisor == 0 and x.x1 > x.x2 + 0.25 else -1

    return LabelingFunction(f"lf_pos_{divisor}", f)


def get_negative_labeling_function(divisor: int) -> LabelingFunction:
    """Get LabelingFunction that abstains unless x0 is divisible by divisor."""

    def f(x):
        return 0 if x.x0 % divisor == 0 and x.x1 <= x.x2 + 0.25 else -1

    return LabelingFunction(f"lf_neg_{divisor}", f)


@preprocessor()
def copy_features(x: DataPoint) -> DataPoint:
    """Compute x2 + 0.25 for direct comparison to x1."""
    x.x3 = x.x2 + 0.25
    return x


@labeling_function(pre=[copy_features], resources=dict(divisor=3))
def f(x: DataPoint, divisor: int) -> int:
    # Abstain unless x0 is divisible by divisor.
    return 0 if x.x0 % divisor == 1 and x.x1 > x.x3 else -1


class LabelingConvergenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure deterministic runs
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        # Create raw data
        cls.N_TRAIN = 1500

        cls.cardinality = 2
        cls.df_train = create_data(cls.N_TRAIN)

    @pytest.mark.complex
    def test_labeling_convergence(self) -> None:
        """Test convergence of end to end labeling pipeline."""
        # Apply LFs
        labeling_functions = (
            [f]
            + [get_positive_labeling_function(divisor) for divisor in range(2, 9)]
            + [get_negative_labeling_function(divisor) for divisor in range(2, 9)]
        )
        applier = PandasLFApplier(labeling_functions)
        L_train = applier.apply(self.df_train, progress_bar=False)

        self.assertEqual(L_train.shape, (self.N_TRAIN, len(labeling_functions)))

        # Train LabelModel
        label_model = LabelModel(cardinality=self.cardinality, verbose=False)
        label_model.fit(L_train, n_epochs=100, lr=0.01, l2=0.0)
        Y_lm = label_model.predict_proba(L_train).argmax(axis=1)
        Y = self.df_train.y
        err = np.where(Y != Y_lm, 1, 0).sum() / self.N_TRAIN
        self.assertLess(err, 0.05)


if __name__ == "__main__":
    unittest.main()
