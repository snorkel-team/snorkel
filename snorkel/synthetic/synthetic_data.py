from functools import partial
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from snorkel.augmentation.tf import LambdaTransformationFunction
from snorkel.labeling.lf import LabelingFunction
from snorkel.types import DataPoint


def generate_mog_dataset(
    n: int, d: int, cov: Optional[np.ndarray] = None, n_noise_dim: int = 0
) -> pd.DataFrame:
    """
    Generates a simple mixture-of-gaussians (MOG) dataset consisting of
    d-dim vectors x \in \mathbb{R}^d, and binary labels y \in {1,2}.
    Returns as a pandas DataFrame
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
    """
    Takes as input a DataPoint x with attribute x \in \mathbb{R}^d, and
    outputs based on the `index`th entry, with probability `abstain_rate` of outputting 0 (abstain).
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
    """
    Generates a list of m labeling functions (LFs) that each abstain randomly
    with probability `abstain_rate`, else label based on the ith entry of
    input DataPoint x.x.
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
    x.x[i] = np.random.rand()
    return x


def generate_resampling_tfs(
    dims: Union[int, List[int]]
) -> List[LambdaTransformationFunction]:
    if isinstance(dims, int):
        dims = list(range(dims))
    return [
        LambdaTransformationFunction(f"tf_{i}", partial(tf_template, i=i)) for i in dims
    ]
