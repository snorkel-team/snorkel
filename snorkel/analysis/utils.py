import random
from typing import Dict, List, Optional, Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set the Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def probs_to_preds(probs: np.ndarray) -> np.ndarray:
    """Convert an array of probabilistic labels into an array of predictions.

    Parameters
    ----------
    prob
        A [num_datapoints, num_classes] array of probabilistic labels such that each
        row sums to 1.

    Returns
    -------
    np.ndarray
        A [num_datapoints, 1] array of predictions (integers in [1, ..., num_classes])
    """
    return np.argmax(probs, axis=1) + 1


def preds_to_probs(preds: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert an array of predictions into an array of probabilistic labels.

    Parameters
    ----------
    pred
        A [num_datapoints] or [num_datapoints, 1] array of predictions

    Returns
    -------
    np.ndarray
        A [num_datapoints, num_classes] array of probabilistic labels with probability
        of 1.0 in the column corresponding to the prediction
    """
    return np.eye(num_classes)[preds.squeeze() - 1]


def to_flattened_int_array(
    X: np.ndarray, flatten: bool = True, cast_to_int: bool = True
) -> np.ndarray:
    """Convert an array to a flattened vector of ints.

    Flatten [n, 1] arrays to [n] and cast all values to ints.
    This method is typically used to sanitize labels before use with analysis tools or
    metrics that expect 1D arrays as inputs.

    Parameters
    ----------
    X
        An array to possibly flatten and possibly cast to int
    flatten
        If True, flatten array into a 1D array
    cast_to_int
        If True, cast all values to ints

    Returns
    -------
    np.ndarray
        The converted array

    Raises
    ------
    ValueError
        Provided input could not be converted to an np.ndarray
    """
    # Correct shape
    if flatten:
        if (X.ndim > 1) and (1 in X.shape):
            X = X.flatten()
        if X.ndim != 1:
            raise ValueError("Input could not be converted to 1d np.array")

    # Convert to ints
    if cast_to_int:
        if np.any(np.not_equal(np.mod(X, 1), 0)):
            raise ValueError("Input contains at least one non-integer value.")
        X = X.astype(np.dtype(int))

    return X


def convert_labels(
    Y: Optional[Union[np.ndarray, torch.Tensor]], source: str, target: str
) -> Union[np.ndarray, torch.Tensor]:
    """Convert a matrix from one label type convention to another.

    The conventions are:

    "categorical": [0: abstain, 1: positive, 2: negative]
    "plusminus": [0: abstain, 1: positive, -1: negative]
    "onezero": [0: negative, 1: positive]

    Parameters
    ----------
    Y
        A np.ndarray or torch.Tensor of labels (ints) using source convention
    source
        The convention the labels are currently expressed in
    target
        The convention to convert the labels to

    Returns
    -------
    ArrayLike
        An np.ndarray or torch.Tensor of labels (ints) using the target convention

    Raises
    ------
    ValueError
        Incorrect input array format
    """
    if Y is None:
        return Y
    if isinstance(Y, np.ndarray):
        Y = Y.copy()
    elif isinstance(Y, torch.Tensor):
        Y = Y.clone()
    else:
        raise ValueError("Unrecognized label data type.")
    negative_map = {"categorical": 2, "plusminus": -1, "onezero": 0}
    Y[Y == negative_map[source]] = negative_map[target]
    return Y


def filter_labels(
    label_dict: Dict[str, np.ndarray], filter_dict: Dict[str, List[int]]
) -> Dict[str, np.ndarray]:
    """Filter out examples from arrays based on specified labels to filter.

    The most common use of this method is to remove examples whose gold label is
    unknown (marked with a 0) or examples whose predictions were abstains (also 0)
    before calculating metrics.

    NB: If an example matches the filter criteria for any label set, it will be removed
    from all label sets (so that the returned arrays are of the same size and still
    aligned).

    Parameters
    ----------
    label_dict
        A mapping from label set name to the array of labels
        The arrays in a label_dict.values() are assumed to be aligned
    filter_dict
        A mapping from label set name to the labels that should be filtered out for
        that label set

    Returns
    -------
    Dict[str, np.ndarray]
        A mapping with the same keys as label_dict but with filtered arrays as values

    Example
    -------
    >>> golds = np.array([0, 1, 1, 2, 2])
    >>> preds = np.array([1, 1, 1, 2, 0])
    >>> filtered = filter_labels(
    ...     label_dict={"golds": golds, "preds": preds},
    ...     filter_dict={"golds": [0], "preds": [0]}
    ... )
    >>> filtered["golds"]
    array([1, 1, 2])
    >>> filtered["preds"]
    array([1, 1, 2])
    """
    masks = []
    for label_name, filter_values in filter_dict.items():
        if label_dict[label_name] is not None:
            masks.append(_get_mask(label_dict[label_name], filter_values))
    mask = np.multiply(*masks) if len(masks) > 1 else masks[0]

    filtered = {}
    for label_name, label_array in label_dict.items():
        filtered[label_name] = label_array[mask] if label_array is not None else None
    return filtered


def _get_mask(label_array: np.ndarray, filter_values: List[int]) -> np.ndarray:
    """Return a boolean mask marking which labels are not in filter_values.

    Parameters
    ----------
    label_array
        An array of labels
    filter_values
        A list of values that should be filtered out of the label array

    Returns
    -------
    np.ndarray
        A boolean mask indicating whether to keep (1) or filter (0) each example
    """
    mask = np.ones_like(label_array).astype(bool)
    for value in filter_values:
        mask *= np.where(label_array != value, 1, 0).astype(bool)

    return mask
