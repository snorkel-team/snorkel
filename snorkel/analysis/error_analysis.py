from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import numpy as np

from snorkel.utils import to_int_label_array


def get_label_buckets(*y: np.ndarray) -> Dict[Tuple[int, ...], np.ndarray]:
    """Return data point indices bucketed by label combinations.

    Parameters
    ----------
    *y
        A list of np.ndarray of (int) labels

    Returns
    -------
    Dict[Tuple[int, ...], np.ndarray]
        A mapping of each label bucket to a NumPy array of its corresponding indices

    Example
    -------
    A common use case is calling ``buckets = label_buckets(Y_gold, Y_pred)`` where
    ``Y_gold`` is a set of gold (i.e. ground truth) labels and ``Y_pred`` is a
    corresponding set of predicted labels.

    >>> Y_gold = np.array([1, 1, 1, 0])
    >>> Y_pred = np.array([1, 1, -1, -1])
    >>> buckets = get_label_buckets(Y_gold, Y_pred)

    The returned ``buckets[(i, j)]`` is a NumPy array of data point indices with
    true label i and predicted label j.

    More generally, the returned indices within each bucket refer to the order of the
    labels that were passed in as function arguments.

    >>> buckets[(1, 1)]  # true positives
    array([0, 1])
    >>> (1, 0) in buckets  # false positives
    False
    >>> (0, 1) in buckets  # false negatives
    False
    >>> (0, 0) in buckets  # true negatives
    False
    >>> buckets[(1, -1)]  # abstained positives
    array([2])
    >>> buckets[(0, -1)]  # abstained negatives
    array([3])
    """
    buckets: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    y_flat = list(map(lambda x: to_int_label_array(x, flatten_vector=True), y))
    if len(set(map(len, y_flat))) != 1:
        raise ValueError("Arrays must all have the same number of elements")
    for i, labels in enumerate(zip(*y_flat)):
        buckets[labels].append(i)
    return {k: np.array(v) for k, v in buckets.items()}
