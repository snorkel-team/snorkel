from typing import Any

import numpy as np
import scipy.sparse as sparse

from snorkel.analysis.utils import arraylike_to_numpy
from snorkel.labeling.model.label_model import LabelModel
from snorkel.types import ArrayLike


class BaselineVoter(LabelModel):
    """Parent baseline label model class with method train_model()."""

    def train_model(self, *args: Any, **kwargs: Any) -> None:
        """Train majority class model.

        Set class balance for majority class label model.

        Parameters
        ----------
        balance
            A [1, k] array of class probabilities
        """
        pass


class RandomVoter(BaselineVoter):
    """Random vote label model.

    Example
    -------
    >>> L = np.array([[1, 1, 0], [0, 1, 2], [2, 0, 1]])
    >>> random_voter = RandomVoter()
    >>> predictions = random_voter.predict_proba(L)
    """

    def predict_proba(self, L: sparse.spmatrix) -> np.ndarray:
        """
        Assign random votes to the data points.

        Parameters
        ----------
        L
            An [n, m] matrix of labels

        Returns
        -------
        np.ndarray
            A [n, k] array of probabilistic labels

        Example
        -------
        >>> L = np.array([[1, 1, 0], [0, 1, 2], [2, 0, 1]])
        >>> random_voter = RandomVoter()
        >>> predictions = random_voter.predict_proba(L)
        """
        n = L.shape[0]
        Y_p = np.random.rand(n, self.cardinality)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p


class MajorityClassVoter(LabelModel):
    """Majority class label model."""

    def train_model(  # type: ignore
        self, balance: ArrayLike, *args: Any, **kwargs: Any
    ) -> None:
        """Train majority class model.

        Set class balance for majority class label model.

        Parameters
        ----------
        balance
            A [1, k] array of class probabilities
        """
        self.balance = np.array(balance)

    def predict_proba(self, L: sparse.spmatrix) -> np.ndarray:
        """Predict probabilities using majority class.

        Assign majority class vote to each datapoint.
        In case of multiple majority classes, assign equal probabilities among them.


        Parameters
        ----------
        L
            An [n, m] matrix of labels

        Returns
        -------
        np.ndarray
            A [n, k] array of probabilistic labels

        Example
        -------
        >>> L = np.array([[1, 1, 0], [0, 1, 2], [2, 0, 1]])
        >>> maj_class_voter = MajorityClassVoter()
        >>> maj_class_voter.train_model(balance=[0.8, 0.2])
        >>> maj_class_voter.predict_proba(L)
        array([[1., 0.],
               [1., 0.],
               [1., 0.]])
        """
        n = L.shape[0]
        Y_p = np.zeros((n, self.cardinality))
        max_classes = np.where(self.balance == max(self.balance))
        for c in max_classes:
            Y_p[:, c] = 1.0
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p


class MajorityLabelVoter(BaselineVoter):
    """Majority vote label model."""

    def predict_proba(self, L: sparse.spmatrix) -> np.ndarray:
        """Predict probabilities using majority vote.

        Assign vote by calculating majority vote across all labeling functions.
        In case of ties, non-integer probabilities are possible.

        Parameters
        ----------
        L
            An [n, m] matrix of labels

        Returns
        -------
        np.ndarray
            A [n, k] array of probabilistic labels

        Example
        -------
        >>> L = np.array([[1, 1, 0], [0, 1, 2], [2, 0, 1]])
        >>> maj_voter = MajorityLabelVoter()
        >>> maj_voter.predict_proba(L)
        array([[1. , 0. ],
               [0.5, 0.5],
               [0.5, 0.5]])
        """
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
