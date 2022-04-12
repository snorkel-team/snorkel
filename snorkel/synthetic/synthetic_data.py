from typing import Tuple

import numpy as np


def generate_simple_label_matrix(
    n: int, m: int, cardinality: int, abstain_multiplier: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    cardinality
        Cardinality of true labels (i.e. not including abstains)
    abstain_multiplier
        Factor to multiply the probability of abstaining by

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the LF conditional probabilities P,
        the true labels Y, and the output label matrix L
    """
    # Generate the conditional probability tables for the LFs
    # The first axis is LF, second is LF output label, third is true class label
    # Note that we include abstains in the LF output space, and that we bias the
    # conditional probabilities towards being non-adversarial
    P = np.empty((m, cardinality + 1, cardinality))
    for i in range(m):
        p = np.random.rand(cardinality + 1, cardinality)

        # Bias the LFs to being non-adversarial
        p[1:, :] += (cardinality - 1) * np.eye(cardinality)

        # Optionally increase the abstain probability by some multiplier; note this is
        # to simulate the common setting where LFs label very sparsely
        p[0, :] *= abstain_multiplier

        # Normalize the conditional probabilities table
        P[i] = p @ np.diag(1 / p.sum(axis=0))

    # Generate the true datapoint labels
    # Note: Assuming balanced classes to start
    Y = np.random.choice(cardinality, n)

    # Generate the label matrix L
    L: np.ndarray = np.empty((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            L[i, j] = np.random.choice(cardinality + 1, p=P[j, :, Y[i]]) - 1
    return P, Y, L
