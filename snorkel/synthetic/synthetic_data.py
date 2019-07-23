from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


def generate_simple_label_matrix(
    n: int, m: int, k: int
) -> Tuple[np.ndarray, np.ndarray, csr_matrix]:
    """Generate a synthetic label matrix with true parameters and labels.

    This function generates a set of labeling function conditional probability tables,
    P(LF=l | Y=y), stored as a matrix P, and true labels Y, and then generates the
    resulting label matrix L.

    Parameters
    ----------
    n
        Number of data points
    m
        Number of labeling functions
    k
        Cardinality of true labels (i.e. not including abstains)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, csr_matrix]
        A tuple containing the LF conditional probabilities P, the true labels Y, and the output label matrix L
    """
    # Generate the conditional probability tables for the LFs
    # The first axis is LF, second is LF output label, third is true class label
    # Note that we include abstains in the LF output space, and that we bias the
    # conditional probabilities towards being non-adversarial
    P = np.zeros((m, k + 1, k))
    for i in range(m):
        p = np.random.rand(k + 1, k)
        p[1:, :] += (k - 1) * np.eye(k)
        P[i] = p @ np.diag(1 / p.sum(axis=0))

    # Generate the true datapoint labels
    # Note: Assuming balanced classes to start
    Y = np.random.choice(k, n) + 1  # Note y \in {1,...,self.k}

    # Generate the label matrix L
    L = lil_matrix((n, m))
    for i in range(n):
        for j in range(m):
            L[i, j] = np.random.choice(k + 1, p=P[j, :, Y[i] - 1])
    L = L.tocsr()
    return P, Y, L
