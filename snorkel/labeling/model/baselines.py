from typing import Any

import numpy as np
import scipy.sparse as sparse

from snorkel.analysis.utils import arraylike_to_numpy
from snorkel.types import ArrayLike

from .label_model import LabelModel


class BaselineVoter(LabelModel):
    def train_model(self, *args: Any, **kwargs: Any) -> None:
        pass


class RandomVoter(BaselineVoter):
    """
    A class that votes randomly among the available labels
    """

    def predict_proba(self, L: sparse.spmatrix) -> np.ndarray:
        """
        Args:
            L: An [n, m] scipy.sparse matrix of labels
        Returns:
            output: A [n, k] np.ndarray of probabilistic labels
        """
        n = L.shape[0]
        Y_p = np.random.rand(n, self.cardinality)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p


class MajorityClassVoter(LabelModel):
    """
    A class that places all probability on the majority class based on class
    balance (and ignoring the label matrix).

    Note that in the case of ties, non-integer probabilities are possible.
    """

    def train_model(  # type: ignore
        self, balance: ArrayLike, *args: Any, **kwargs: Any
    ) -> None:
        """
        Args:
            balance: A 1d arraylike that sums to 1, corresponding to the
                (possibly estimated) class balance.
        """
        self.balance = np.array(balance)

    def predict_proba(self, L: sparse.spmatrix) -> np.ndarray:
        n = L.shape[0]
        Y_p = np.zeros((n, self.cardinality))
        max_classes = np.where(self.balance == max(self.balance))
        for c in max_classes:
            Y_p[:, c] = 1.0
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p


class MajorityLabelVoter(BaselineVoter):
    """
    A class that places all probability on the majority label from all
    non-abstaining LFs for that task.

    Note that in the case of ties, non-integer probabilities are possible.
    """

    def predict_proba(self, L: sparse.spmatrix) -> np.ndarray:
        L = arraylike_to_numpy(L, flatten=False)
        n, m = L.shape
        Y_p = np.zeros((n, self.cardinality))
        for i in range(n):
            counts = np.zeros(self.cardinality)
            for j in range(m):
                if L[i, j]:
                    counts[L[i, j] - 1] += 1
            Y_p[i, :] = np.where(counts == max(counts), 1, 0)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p
