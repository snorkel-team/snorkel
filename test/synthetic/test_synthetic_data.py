import unittest
from typing import Optional

import numpy as np

from snorkel.labeling import LFAnalysis
from snorkel.synthetic.synthetic_data import generate_simple_label_matrix


class TestGenerateSimpleLabelMatrix(unittest.TestCase):
    """Testing the generate_simple_label_matrix function."""

    def setUp(self) -> None:
        """Set constants for the tests."""
        self.m = 10  # Number of LFs
        self.n = 1000  # Number of data points

    def _test_generate_L(self, k: int, decimal: Optional[int] = 2) -> None:
        """Test generated label matrix L for consistency with P, Y.

        This tests for consistency between the true conditional LF probabilities, P,
        and the empirical ones computed from L and Y, where P, L, and Y are generated
        by the generate_simple_label_matrix function.

        Parameters
        ----------
        k
            Cardinality
        decimal
            Number of decimals to check element-wise error, err < 1.5 * 10**(-decimal)
        """
        np.random.seed(123)
        P, Y, L = generate_simple_label_matrix(self.n, self.m, k)
        P_emp = LFAnalysis(L).lf_empirical_probs(Y, k=k)
        np.testing.assert_array_almost_equal(P, P_emp, decimal=decimal)

    def test_generate_L(self) -> None:
        """Test the generated dataset for consistency."""
        self._test_generate_L(2, decimal=1)

    def test_generate_L_multiclass(self) -> None:
        """Test the generated dataset for consistency with cardinality=3."""
        self._test_generate_L(3, decimal=1)


if __name__ == "__main__":
    unittest.main()
