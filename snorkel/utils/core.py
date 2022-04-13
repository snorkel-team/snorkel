import hashlib
from typing import Dict, List, Optional

import numpy as np


def _hash(i: int) -> int:
    """Deterministic hash function."""
    byte_string = str(i).encode("utf-8")
    return int(hashlib.sha1(byte_string).hexdigest(), 16)


def probs_to_preds(
    probs: np.ndarray, tie_break_policy: str = "random", tol: float = 1e-5
) -> np.ndarray:
    """Convert an array of probabilistic labels into an array of predictions.

    Policies to break ties include:
    "abstain": return an abstain vote (-1)
    "true-random": randomly choose among the tied options
    "random": randomly choose among tied option using deterministic hash

    NOTE: if tie_break_policy="true-random", repeated runs may have slightly different results due to difference in broken ties

    Parameters
    ----------
    prob
        A [num_datapoints, num_classes] array of probabilistic labels such that each
        row sums to 1.
    tie_break_policy
        Policy to break ties when converting probabilistic labels to predictions
    tol
        The minimum difference among probabilities to be considered a tie

    Returns
    -------
    np.ndarray
        A [n] array of predictions (integers in [0, ..., num_classes - 1])

    Examples
    --------
    >>> probs_to_preds(np.array([[0.5, 0.5, 0.5]]), tie_break_policy="abstain")
    array([-1])
    >>> probs_to_preds(np.array([[0.8, 0.1, 0.1]]))
    array([0])
    """
    num_datapoints, num_classes = probs.shape
    if num_classes <= 1:
        raise ValueError(
            f"probs must have probabilities for at least 2 classes. "
            f"Instead, got {num_classes} classes."
        )

    Y_pred = np.empty(num_datapoints)
    diffs = np.abs(probs - probs.max(axis=1).reshape(-1, 1))

    for i in range(num_datapoints):
        max_idxs = np.where(diffs[i, :] < tol)[0]
        if len(max_idxs) == 1:
            Y_pred[i] = max_idxs[0]
        # Deal with "tie votes" according to the specified policy
        elif tie_break_policy == "random":
            Y_pred[i] = max_idxs[_hash(i) % len(max_idxs)]
        elif tie_break_policy == "true-random":
            Y_pred[i] = np.random.choice(max_idxs)
        elif tie_break_policy == "abstain":
            Y_pred[i] = -1
        else:
            raise ValueError(
                f"tie_break_policy={tie_break_policy} policy not recognized."
            )
    return Y_pred.astype(np.int_)


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
    if np.any(preds < 0):
        raise ValueError("Could not convert abstained vote to probability")
    return np.eye(num_classes)[preds.squeeze()]


def to_int_label_array(X: np.ndarray, flatten_vector: bool = True) -> np.ndarray:
    """Convert an array to a (possibly flattened) array of ints.

    Cast all values to ints and possibly flatten [n, 1] arrays to [n].
    This method is typically used to sanitize labels before use with analysis tools or
    metrics that expect 1D arrays as inputs.

    Parameters
    ----------
    X
        An array to possibly flatten and possibly cast to int
    flatten_vector
        If True, flatten array into a 1D array

    Returns
    -------
    np.ndarray
        The converted array

    Raises
    ------
    ValueError
        Provided input could not be converted to an np.ndarray
    """
    if np.any(np.not_equal(np.mod(X, 1), 0)):
        raise ValueError("Input contains at least one non-integer value.")
    X = X.astype(np.dtype(int))
    # Correct shape
    if flatten_vector:
        X = X.squeeze()
        if X.ndim == 0:
            X = np.expand_dims(X, 0)
        if X.ndim != 1:
            raise ValueError("Input could not be converted to 1d np.array")
    return X


def filter_labels(
    label_dict: Dict[str, Optional[np.ndarray]], filter_dict: Dict[str, List[int]]
) -> Dict[str, np.ndarray]:
    """Filter out examples from arrays based on specified labels to filter.

    The most common use of this method is to remove examples whose gold label is
    unknown (marked with a -1) or examples whose predictions were abstains (also -1)
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
    >>> golds = np.array([-1, 0, 0, 1, 0])
    >>> preds = np.array([0, 0, 0, 1, -1])
    >>> filtered = filter_labels(
    ...     label_dict={"golds": golds, "preds": preds},
    ...     filter_dict={"golds": [-1], "preds": [-1]}
    ... )
    >>> filtered["golds"]
    array([0, 0, 1])
    >>> filtered["preds"]
    array([0, 0, 1])
    """
    masks = []
    for label_name, filter_values in filter_dict.items():
        label_array: Optional[np.ndarray] = label_dict.get(label_name)
        if label_array is not None:
            # _get_mask requires not-null input
            masks.append(_get_mask(label_array, filter_values))
    mask = (np.multiply(*masks) if len(masks) > 1 else masks[0]).squeeze()

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
    mask: np.ndarray = np.ones_like(label_array).astype(bool)
    for value in filter_values:
        mask *= np.where(label_array != value, 1, 0).astype(bool)
    return mask
