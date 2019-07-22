from collections import defaultdict
from typing import Dict, DefaultDict, List, Tuple

import numpy as np

from .utils import to_int_label_array


Coord = Tuple[int, int]


def error_buckets(golds: np.ndarray, preds: np.ndarray) -> Dict[Coord, List[int]]:
    """Return examples (or their indices) bucketed by gold label/pred label combination.

    Returned buckets[i,j] is a list of items with predicted label i and true label j.
    For a binary problem with (1=positive, 2=negative):
        buckets[1,1] = true positives
        buckets[1,2] = false positives
        buckets[2,1] = false negatives
        buckets[2,2] = true negatives

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
    buckets: DefaultDict[Coord, List[int]] = defaultdict(list)
    golds = to_int_label_array(golds)
    preds = to_int_label_array(preds)
    for i, (l, y) in enumerate(zip(preds, golds)):
        buckets[(l, y)].append(i)
    return dict(buckets)
