from typing import Any

import numpy as np

from snorkel.labeling.model.base_labeler import BaseLabeler


class RandomVoter(BaseLabeler):
    """Random vote label model.

    Example
    -------
    >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
    >>> random_voter = RandomVoter()
    >>> predictions = random_voter.predict_proba(L)
    """

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
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
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> random_voter = RandomVoter()
        >>> predictions = random_voter.predict_proba(L)
        """
        n = L.shape[0]
        Y_p = np.random.rand(n, self.cardinality)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p


class MajorityClassVoter(BaseLabeler):
    """Majority class label model."""

    def fit(  # type: ignore
        self, balance: np.ndarray, *args: Any, **kwargs: Any
    ) -> None:
        """Train majority class model.

        Set class balance for majority class label model.

        Parameters
        ----------
        balance
            A [k] array of class probabilities
        """
        self.balance = balance

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
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
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> maj_class_voter = MajorityClassVoter()
        >>> maj_class_voter.fit(balance=np.array([0.8, 0.2]))
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


class MajorityLabelVoter(BaseLabeler):
    """Majority vote label model."""

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
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
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> maj_voter = MajorityLabelVoter()
        >>> maj_voter.predict_proba(L)
        array([[1. , 0. ],
               [0.5, 0.5],
               [0.5, 0.5]])
        """
        n, m = L.shape
        Y_p = np.zeros((n, self.cardinality))
        for i in range(n):
            counts = np.zeros(self.cardinality)
            for j in range(m):
                if L[i, j] != -1:
                    counts[L[i, j]] += 1
            Y_p[i, :] = np.where(counts == max(counts), 1, 0)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p
