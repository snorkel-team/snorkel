from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import numpy as np

from .utils import to_int_label_array


def label_buckets(*y: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
    """Return data point indices bucketed by label combinations.

    A common use case is calling ``label_buckets(Y_gold, Y_pred)`` where
    ``Y_gold`` is a set of gold (i.e. ground truth) labels and ``Y_pred`` is a
    corresponding set of predicted labels.
    The returned ``buckets[(i, j)]`` is a NumPy array of data point indices with
    predicted label i and true label j.
    For a binary problem with (1 = positive, 0 = negative):
        ``buckets[(1, 1)]`` = true positives
        ``buckets[(1, 0)]`` = false negatives
        ``buckets[(0, 1)]`` = false positives
        ``buckets[(0, 0)]`` = true negatives

    Parameters
    ----------
    *y
        A list of np.ndarray of (int) labels

    Returns
    -------
    Dict
        A mapping of each label bucket to a NumPy array of its corresponding indices
    """
    buckets: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    y_flat = map(lambda x: to_int_label_array(x, flatten_vector=True), y)
    for i, labels in enumerate(zip(*y_flat)):
        buckets[labels].append(i)
    return {k: np.array(v) for k, v in buckets.items()}
