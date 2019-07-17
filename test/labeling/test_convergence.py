import unittest

import numpy as np
import pandas as pd
import pytest

from snorkel.analysis.utils import set_seed
from snorkel.labeling.apply.pandas import PandasLFApplier
from snorkel.labeling.lf import LabelingFunction, labeling_function
from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling.preprocess import preprocessor
from snorkel.types import DataPoint


class LabelingConvergenceTest(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        # Ensure deterministic runs
        set_seed(123)

        # Create raw data
        cls.N_TRAIN = 1500
        cls.N_VALID = 300

        cls.k = 2
        cls.df_train = create_data(cls.N_TRAIN)
        cls.df_valid = create_data(cls.N_VALID)

    @pytest.mark.complex
    def test_labeling_convergence(self) -> None:
        """Test convergence of end to end labeling pipeline."""
        # Apply LFs
        labeling_functions = [f] + [
            get_labeling_function(divisor) for divisor in range(2, 6)
        ]
        applier = PandasLFApplier(labeling_functions)
        L_train = applier.apply(self.df_train)
        L_valid = applier.apply(self.df_valid)

        self.assertEqual(L_train.shape, (self.N_TRAIN, len(labeling_functions)))
        self.assertEqual(L_valid.shape, (self.N_VALID, len(labeling_functions)))

        # Train LabelModel
        label_model = LabelModel(cardinality=self.k, verbose=False)
        label_model.train_model(L_train, lr=0.01, l2=0.0, n_epochs=100)
        Y_lm = label_model.predict_proba(L_train.todense()).argmax(axis=1) + 1
        Y = self.df_train.y
        err = np.where(Y != Y_lm, 1, 0).sum() / self.N_TRAIN
        self.assertLess(err, 0.1)


def create_data(n: int) -> pd.DataFrame:
    X = np.random.random((n, 2)) * 2 - 1
    Y = (X[:, 0] > X[:, 1] + 0.25).astype(int) + 1

    df = pd.DataFrame({"idx": np.arange(n), "x1": X[:, 0], "x2": X[:, 1], "y": Y})
    return df


def get_labeling_function(divisor: int) -> LabelingFunction:
    """Get LabelingFunction that abstains unless idx is divisible by divisor."""

    def f(x: DataPoint) -> int:
        if x.idx % divisor != 0:
            return 0
        if x.x1 > x.x2 + 0.25:
            return 2
        else:
            return 1

    return LabelingFunction(f"lf_{divisor}", f)


@preprocessor()
def copy_features(x: DataPoint) -> DataPoint:
    """Compute x2 + 0.25 for direct comparison to x1."""
    x.x3 = x.x2 + 0.25
    return x


@labeling_function(preprocessors=[copy_features], resources=dict(divisor=10))
def f(x: DataPoint, divisor: int) -> int:
    # Abstain unless idx is divisible by divisor.
    if x.idx % divisor != 0:
        return 0
    if x.x1 > x.x3:
        return 2
    else:
        return 1


if __name__ == "__main__":
    unittest.main()
