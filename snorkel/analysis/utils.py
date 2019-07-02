from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sparse
import torch

from snorkel.types import ArrayLike


def probs_to_preds(probs: np.ndarray) -> np.ndarray:
    """Convert an array of probabilistic labels into an array of predictions

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
    """Convert an array of probabilistic labels into an array of predictions

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


def arraylike_to_numpy(
    array_like: ArrayLike, flatten: bool = True, cast_to_int: bool = True
) -> np.ndarray:
    """Convert an ArrayLike (e.g., list, tensor, etc.) to a numpy array

    Also optionally flatten [n, 1] arrays to [n] and cast all values to ints.
    This method is typically used to sanitize labels before use with analysis tools or
    metrics that expect 1D numpy arrays as inputs.

    Parameters
    ----------
    array_like
        An Arraylike to convert
    flatten
        If True, flatten numpy array into a 1D array
    cast_to_int
        If True, cast all values to ints

    Returns
    -------
    np.ndarray
        The input converted to an np.ndarray

    Raises
    ------
    ValueError
        Provided input could not be converted to an np.ndarray
    """

    orig_type = type(array_like)

    # Convert to np.ndarray
    if isinstance(array_like, np.ndarray):
        pass
    elif isinstance(array_like, list):
        array_like = np.array(array_like)
    elif isinstance(array_like, sparse.spmatrix):
        array_like = array_like.toarray()
    elif isinstance(array_like, torch.Tensor):
        array_like = array_like.numpy()
    elif not isinstance(array_like, np.ndarray):
        array_like = np.array(array_like)
    else:
        msg = f"Input of type {orig_type} could not be converted to an np.ndarray"
        raise ValueError(msg)

    # Correct shape
    if flatten:
        if (array_like.ndim > 1) and (1 in array_like.shape):
            array_like = array_like.flatten()
        if array_like.ndim != 1:
            raise ValueError("Input could not be converted to 1d np.array")

    # Convert to ints
    if cast_to_int:
        if any(array_like % 1):
            raise ValueError("Input contains at least one non-integer value.")
        array_like = array_like.astype(np.dtype(int))

    return array_like


def convert_labels(Y: ArrayLike, source: str, target: str) -> ArrayLike:
    """Convert a matrix from one label type to another

    Args:
        Y: A np.ndarray or torch.Tensor of labels (ints) using source convention
        source: The convention the labels are currently expressed in
        target: The convention to convert the labels to
    Returns:
        Y: an np.ndarray or torch.Tensor of labels (ints) using the target convention

    Conventions:
        'categorical': [0: abstain, 1: positive, 2: negative]
        'plusminus': [0: abstain, 1: positive, -1: negative]
        'onezero': [0: negative, 1: positive]

    Note that converting to 'onezero' will combine abstain and negative labels.
    """
    if Y is None:
        return Y
    if isinstance(Y, np.ndarray):
        Y = Y.copy()
        assert Y.dtype == np.int64
    elif isinstance(Y, torch.Tensor):
        Y = Y.clone()
        assert isinstance(Y, torch.LongTensor)
    else:
        raise ValueError("Unrecognized label data type.")
    negative_map = {"categorical": 2, "plusminus": -1, "onezero": 0}
    Y[Y == negative_map[source]] = negative_map[target]
    return Y


def filter_labels(
    golds: ArrayLike,
    preds: ArrayLike,
    probs: Optional[ArrayLike] = None,
    ignore_in_golds: List[int] = [],
    ignore_in_preds: List[int] = [],
) -> Tuple[ArrayLike, ArrayLike, Optional[ArrayLike]]:
    """Converts golds, preds, and probs to `np.ndarray`s and filters out examples

    Parameters
    ----------
    golds
        Gold labels [n_datapoints, 1]
    preds
        Prediction labels [n_datapoints, 1]
    probs
        Probablistic labels [n_datapoints, n_classes]
    ignore_in_golds
        A list of integer gold labels corresponding to examples to filter out
    ignore_in_preds
        A list of integer prediction labels corresponding to examples to filter out

    Returns
    -------
    Tuple[
        Union[np.ndarray, None],
        Union[np.ndarray, None],
        Union[np.ndarray, None]
    ]
        Filtered versions of golds, preds, probs
    """
    golds = arraylike_to_numpy(golds) if golds is not None else None
    preds = arraylike_to_numpy(preds) if preds is not None else None
    probs = (
        arraylike_to_numpy(probs, flatten=False, cast_to_int=False)
        if probs is not None
        else None
    )

    mask = _get_mask(golds, preds, ignore_in_golds, ignore_in_preds)
    golds = golds[mask] if golds is not None else None
    preds = preds[mask] if preds is not None else None
    probs = probs[mask] if probs is not None else None

    return golds, preds, probs


def _get_mask(
    golds: Union[np.ndarray, None],
    preds: Union[np.ndarray, None],
    ignore_in_golds: List[int] = [],
    ignore_in_preds: List[int] = [],
) -> np.ndarray:
    """Return a boolean mask for which examples to keep/filter based on user args

    Parameters
    ----------
    golds
        Gold labels [n_datapoints, 1]
    preds
        Prediction labels [n_datapoints, 1]
    ignore_in_golds
        A list of integer gold labels corresponding to examples to filter out
    ignore_in_preds
        A list of integer prediction labels corresponding to examples to filter out

    Returns
    -------
    np.ndarray
        A boolean mask indicating whether to keep (1) or filter (0) each example
    """
    mask = np.ones_like(golds).astype(bool)

    if golds is not None:
        for x in ignore_in_golds:
            mask *= np.where(golds != x, 1, 0).astype(bool)

    if preds is not None:
        for x in ignore_in_preds:
            mask *= np.where(preds != x, 1, 0).astype(bool)

    return mask
