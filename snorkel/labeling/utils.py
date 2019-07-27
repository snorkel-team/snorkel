from typing import Tuple

import numpy as np
import pandas as pd


def filter_unlabeled_dataframe(
    X: pd.DataFrame, y: np.ndarray, L: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Filter out examples not covered by any labeling function.

    Parameters
    ----------
    X
        Data points in a Pandas DataFrame.
    y
        Matrix of probabilities output by label model's predict_proba method.
    L
        Matrix of labels emitted by LFs.

    Returns
    -------
    pd.DataFrame
        Data points that were labeled by at least one LF in L.
    np.ndarray
        Probabilities matrix for data points labeled by at least one LF in L.
    """
    mask = (L != -1).any(axis=1)
    return X.iloc[mask], y[mask]
