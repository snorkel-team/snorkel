from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

from snorkel.augmentation.tf import LambdaTransformationFunction
from snorkel.labeling.lf import LabelingFunction
from snorkel.types import DataPoint


def generate_simple_label_matrix(
    n: int, m: int, cardinality: int
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
    cardinality
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
    P = np.zeros((m, cardinality + 1, cardinality))
    for i in range(m):
        p = np.random.rand(cardinality + 1, cardinality)
        p[1:, :] += (cardinality - 1) * np.eye(cardinality)
        P[i] = p @ np.diag(1 / p.sum(axis=0))

    # Generate the true datapoint labels
    # Note: Assuming balanced classes to start
    Y = np.random.choice(cardinality, n) + 1  # Note y \in {1,...,self.cardinality}

    # Generate the label matrix L
    L = lil_matrix((n, m))
    for i in range(n):
        for j in range(m):
            L[i, j] = np.random.choice(cardinality + 1, p=P[j, :, Y[i] - 1])
    L = L.tocsr()
    return P, Y, L


def generate_mog_dataset(
    n: int, d: int, cov: Optional[np.ndarray] = None, n_noise_dim: int = 0
) -> pd.DataFrame:
    r"""Generate a simple mixture-of-gaussians (MOG) dataset.

    Consists of d-dim vectors x \in \mathbb{R}^d, and binary labels y \in {1,2}.
    Returns as a pandas DataFrame.
    """
    ones = np.ones(d)
    nh = int(np.floor(n / 2))
    if cov is None:
        cov = np.diag(np.random.random(d))
    X_pos = np.random.multivariate_normal(ones, cov, nh)
    X_neg = np.random.multivariate_normal(-1 * ones, cov, n - nh)

    # Combine and shuffle
    X = np.vstack([X_pos, X_neg])
    Y = np.concatenate([np.ones(nh), 2 * np.ones(n - nh)]).astype(int)
    order = list(range(n))
    np.random.shuffle(order)
    X = X[order]
    Y = Y[order]

    # Add some noise
    if n_noise_dim > 0:
        X = np.hstack((X, np.random.rand(X.shape[0], n_noise_dim)))

    # Convert to DataFrame
    return pd.DataFrame(dict(x=list(X), y=Y))


def lf_template(x: DataPoint, index: int = 0, abstain_rate: float = 0.0) -> int:
    r"""LF templates that votes based on attribute x ``index``.

    Takes as input a DataPoint x with attribute x \in \mathbb{R}^d, and outputs
    based on the ``index``th entry, with probability ``abstain_rate`` of outputting 0 (abstain).
    """
    if np.random.random() < abstain_rate:
        return 0
    elif x.x[index] > 0:
        return 1
    else:
        return 2


def generate_single_feature_lfs(
    dims: Union[int, List[int]], abstain_rate: float = 0.0
) -> List[LabelingFunction]:
    """Generate a list of m labeling functions (LFs) based on a single attribute 'x'.

    Each LF abstains with probability ``abstain_rate``, else label based on the
    ith entry of the input DataPoint x.x.
    """
    if isinstance(dims, int):
        dims = list(range(dims))
    return [
        LabelingFunction(
            f"LF_feature_{i}", partial(lf_template, index=i, abstain_rate=abstain_rate)
        )
        for i in dims
    ]


def tf_template(x: DataPoint, i: int) -> DataPoint:
    r"""TF template that transforms data attribute x.x randomly.

    Takes as input a DataPoint x with attribute x \in \mathbb{R}^d and randomly
    transforms ``index``th value.
    """
    x.x[i] = np.random.rand()
    return x


def generate_resampling_tfs(
    dims: Union[int, List[int]]
) -> List[LambdaTransformationFunction]:
    """Generate a list of transformation functions (TFs) that transform attribute 'x'."""
    if isinstance(dims, int):
        dims = list(range(dims))
    return [
        LambdaTransformationFunction(f"tf_{i}", partial(tf_template, i=i)) for i in dims
    ]
