from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import numpy as np

from .utils import to_int_label_array


def error_buckets(
    golds: np.ndarray, preds: np.ndarray
) -> Dict[Tuple[int, int], List[int]]:
    """Return data point indices bucketed by gold label/pred label combination.

    Returned buckets[i,j] is a list of items with predicted label i and true label j.
    For a binary problem with (1=positive, 0=negative):
        buckets[1,1] = true positives
        buckets[1,0] = false positives
        buckets[0,1] = false negatives
        buckets[0,0] = true negatives

    Parameters
    ----------
    golds
        An np.ndarray of gold (int) labels
    preds
        An np.ndarray of (int) predictions

    Returns
    -------
    Dict
        A mapping of each error bucket to its corresponding indices
    """
    buckets: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    golds = to_int_label_array(golds)
    preds = to_int_label_array(preds)
    for i, (l, y) in enumerate(zip(preds, golds)):
        buckets[(l, y)].append(i)
    return dict(buckets)
