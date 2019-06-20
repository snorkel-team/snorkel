from functools import partial
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

from snorkel.labeling.lf import LabelingFunction
from snorkel.types import DataPoint


def generate_mog_dataset(
    n: int, d: int, cov: Optional[np.ndarray] = None
) -> List[DataPoint]:
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

    # Convert to list of objects with x, y attributes
    data = [SimpleNamespace(x=X[i, :], y=Y[i]) for i in range(X.shape[0])]
    return data


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


def generate_single_feature_LFs(
    m: int, abstain_rate: float = 0.0
) -> List[LabelingFunction]:
    """
    Generates a list of m labeling functions (LFs) that each abstain randomly
    with probability `abstain_rate`, else label based on the ith entry of
    input DataPoint x.x.
    """
    lfs = []
    for i in range(m):
        lfs.append(
            LabelingFunction(
                f"LF_feature_{i}",
                partial(lf_template, index=i, abstain_rate=abstain_rate),
            )
        )
    return lfs
