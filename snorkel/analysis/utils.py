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
    return np.argmax(prob, axis=1) + 1


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
    return np.eye(num_classes)[pred.squeeze() - 1]


def arraylike_to_numpy(array_like: ArrayLike) -> np.ndarray:
    """Convert a 1d array-like (e.g,. list, tensor, etc.) to an np.ndarray"""

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
        msg = f"Input of type {orig_type} could not be converted to 1d " "np.ndarray"
        raise ValueError(msg)

    # Correct shape
    if (array_like.ndim > 1) and (1 in array_like.shape):
        array_like = array_like.flatten()
    if array_like.ndim != 1:
        raise ValueError("Input could not be converted to 1d np.array")

    # Convert to ints
    if any(array_like % 1):
        raise ValueError("Input contains at least one non-integer value.")
    array_like = array_like.astype(np.dtype(int))

    return array_like


def convert_labels(Y: ArrayLike, source: str, target: str):
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


def plusminus_to_categorical(Y):
    return convert_labels(Y, "plusminus", "categorical")


def categorical_to_plusminus(Y):
    return convert_labels(Y, "categorical", "plusminus")
