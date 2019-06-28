import unittest

import numpy as np
import torch

from snorkel.labeling.model.class_balance import ClassBalanceModel
from snorkel.model.utils import set_seed


def generate_class_balance(k: int) -> np.ndarray:
    """Generate class balance"""
    p_Y = np.random.random(k)
    p_Y /= p_Y.sum()
    return p_Y


def generate_cond_probs(
    k: int, m: int, bias_diag: bool = True, abstains: bool = False
) -> np.ndarray:
    """Generate conditional probability tables for the m conditionally ind.
    LFs, such that:
        cpts[i, y1, y2] = P(\lambda_i = y1 | Y = y2)

    Args:
        k: (int) Number of classes
        m: (int) Number of LFs
        bias_diag: (bool) If True, adds a bias (proportional to (k-1)) to
            the diagonal of the randomly generated conditional probability
            tables, to enforce assumption that LFs are better than random
        abstains: (bool) Incorporate abstains

    Outputs:
        C: (np.array) An (m, k, k) tensor, if abstains=False; or, if
            abstains=True, (m, k+1, k)
    """
    cpts = []
    k_lf = k + 1 if abstains else k
    for i in range(m):
        a = np.random.random((k_lf, k))
        if bias_diag:
            if abstains:
                a[1:, :] += (k - 1) * np.eye(k)
            else:
                a += (k - 1) * np.eye(k)
        cpts.append(a @ np.diag(1 / a.sum(axis=0)))
    return np.array(cpts)


def generate_L(
    p_Y: np.ndarray, C: np.ndarray, n: int, abstains: bool = False
) -> np.ndarray:
    """Generate a label matrix L, with entries in {0,1,...,k} if
    abstains=True, else in {1,...,k}, given the true class balance, p_Y, and
    a conditional probabilities table C of m cond. ind. LFs"""
    k = len(p_Y)
    m = C.shape[0]

    # Generate true data labels for n data points
    Y = np.random.choice(range(1, k + 1), n, p=p_Y)

    # Generate label matrix L with entries in {0,1,...,k} if abstains=True,
    # else in {1,...,k}
    lf_labels = range(0 if abstains else 1, k + 1)
    L = np.zeros((n, m))
    for j in range(m):
        for y in range(k):
            c_jy = C[j, :, y]
            mask = Y == (y + 1)
            if mask.sum() > 0:
                L[mask, j] = np.random.choice(lf_labels, size=mask.sum(), p=c_jy)
    return L


class ClassBalanceModelTest(unittest.TestCase):
    def _test_model(self, model, p_Y, C, O=None, L=None, tol=1e-3):
        model.train_model(O=O, L=L)
        self.assertLess(np.mean(np.abs(p_Y - model.class_balance)), tol)
        self.assertLess(np.mean(np.abs(C - model.cond_probs)), tol)

    def _test_class_balance_estimation(self, k, m, abstains=False):
        model = ClassBalanceModel(k, abstains=abstains)
        p_Y = generate_class_balance(k)
        C = generate_cond_probs(k, m, bias_diag=True, abstains=abstains)

        # Compute O; mask out diagonal entries
        mask = model.get_mask(m)
        O = np.einsum("aby,cdy,efy,y->acebdf", C, C, C, p_Y)
        O = torch.from_numpy(O).float()
        O[1 - mask] = 0

        # Test recovery of the class balance
        self._test_model(model, p_Y, C, O=O)

    def _test_class_balance_estimation_noisy(self, k, m, n, abstains=False):
        model = ClassBalanceModel(k, abstains=abstains)
        p_Y = generate_class_balance(k)
        C = generate_cond_probs(k, m, bias_diag=True, abstains=abstains)

        # Generate label matrix L
        L = generate_L(p_Y, C, n, abstains=abstains)

        # Test recovery of the class balance
        self._test_model(model, p_Y, C, L=L, tol=1e-2)

    def test_class_balance_estimation_2(self):
        set_seed(123)
        self._test_class_balance_estimation(2, 25)

    def test_class_balance_estimation_3(self):
        set_seed(123)
        self._test_class_balance_estimation(3, 25)

    # Note: This should pass! However, commented out because too slow...
    # def test_class_balance_estimation_5(self):
    #     set_seed(123)
    #     self._test_class_balance_estimation(5, 25)

    def test_class_balance_estimation_2_abstains(self):
        set_seed(123)
        self._test_class_balance_estimation(2, 25, abstains=True)

    def test_class_balance_estimation_2_noisy(self):
        set_seed(123)
        self._test_class_balance_estimation_noisy(2, 25, 5000, abstains=True)


if __name__ == "__main__":
    unittest.main()
