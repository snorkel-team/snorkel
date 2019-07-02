import numpy as np

from .label_model import LabelModel


class BaselineVoter(LabelModel):
    def train_model(self, *args, **kwargs):
        pass


class RandomVoter(BaselineVoter):
    """
    A class that votes randomly among the available labels
    """

    def predict_proba(self, L):
        """
        Args:
            L: An [n, m] scipy.sparse matrix of labels
        Returns:
            output: A [n, k] np.ndarray of probabilistic labels
        """
        n = L.shape[0]
        Y_p = np.random.rand(n, self.k)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p


class MajorityClassVoter(LabelModel):
    """
    A class that places all probability on the majority class based on class
    balance (and ignoring the label matrix).

    Note that in the case of ties, non-integer probabilities are possible.
    """

    def train_model(self, balance, *args, **kwargs):
        """
        Args:
            balance: A 1d arraylike that sums to 1, corresponding to the
                (possibly estimated) class balance.
        """
        self.balance = np.array(balance)

    def predict_proba(self, L):
        n = L.shape[0]
        Y_p = np.zeros((n, self.k))
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

    def predict_proba(self, L):
        L = self._to_numpy(L).astype(int)
        n, m = L.shape
        Y_p = np.zeros((n, self.k))
        for i in range(n):
            counts = np.zeros(self.k)
            for j in range(m):
                if L[i, j]:
                    counts[L[i, j] - 1] += 1
            Y_p[i, :] = np.where(counts == max(counts), 1, 0)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p