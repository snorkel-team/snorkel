from collections import Counter, defaultdict
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from snorkel.types import ArrayLike

from .utils import arraylike_to_numpy


def error_buckets(
    gold: ArrayLike, pred: ArrayLike, X: Optional[Sequence[Any]] = None
) -> Mapping[Tuple[int, int], Any]:
    """Group items by error buckets

    Args:
        gold: an array-like of gold labels (ints)
        pred: an array-like of predictions (ints)
        X: an iterable of items
    Returns:
        buckets: A dict of items where buckets[i,j] is a list of items with
            predicted label i and true label j. If X is None, return indices
            instead.

    For a binary problem with (1=positive, 2=negative):
        buckets[1,1] = true positives
        buckets[1,2] = false positives
        buckets[2,1] = false negatives
        buckets[2,2] = true negatives
    """
    buckets: Mapping[Tuple[int, int], List[Any]] = defaultdict(list)
    gold = arraylike_to_numpy(gold)
    pred = arraylike_to_numpy(pred)
    for i, (y, l) in enumerate(zip(pred, gold)):
        buckets[y, l].append(X[i] if X is not None else i)
    return dict(buckets)


def confusion_matrix(
    gold, pred, null_pred=False, null_gold=False, normalize=False, pretty_print=True
):
    """A shortcut method for building a confusion matrix all at once.

    Args:
        gold: an array-like of gold labels (ints)
        pred: an array-like of predictions (ints)
        null_pred: If True, include the row corresponding to null predictions
        null_gold: If True, include the col corresponding to null gold labels
        normalize: if True, divide counts by the total number of items
        pretty_print: if True, pretty-print the matrix before returning
    """
    conf = ConfusionMatrix(null_pred=null_pred, null_gold=null_gold)
    gold = arraylike_to_numpy(gold)
    pred = arraylike_to_numpy(pred)
    conf.add(gold, pred)
    mat = conf.compile()

    if normalize:
        mat = mat / len(gold)

    if pretty_print:
        conf.display(normalize=normalize)

    return mat


class ConfusionMatrix(object):
    """
    An iteratively built abstention-aware confusion matrix with pretty printing

    Assumed axes are true label on top, predictions on the side.
    """

    def __init__(self, null_pred=False, null_gold=False):
        """
        Args:
            null_pred: If True, include the row corresponding to null
                predictions
            null_gold: If True, include the col corresponding to null gold
                labels

        """
        self.counter = Counter()
        self.mat = None
        self.null_pred = null_pred
        self.null_gold = null_gold

    def __repr__(self):
        if self.mat is None:
            self.compile()
        return str(self.mat)

    def add(self, gold, pred):
        """
        Args:
            gold: a np.ndarray of gold labels (ints)
            pred: a np.ndarray of predictions (ints)
        """
        self.counter.update(zip(gold, pred))

    def compile(self, trim=True):
        k = max([max(tup) for tup in self.counter.keys()]) + 1  # include 0

        mat = np.zeros((k, k), dtype=int)
        for (y, l), v in self.counter.items():
            mat[l, y] = v

        if trim and not self.null_pred:
            mat = mat[1:, :]
        if trim and not self.null_gold:
            mat = mat[:, 1:]

        self.mat = mat
        return mat

    def display(self, normalize=False, indent=0, spacing=2, decimals=3, mark_diag=True):
        mat = self.compile(trim=False)
        m, n = mat.shape
        tab = " " * spacing
        margin = " " * indent

        # Print headers
        s = margin + " " * (5 + spacing)
        for j in range(n):
            if j == 0 and not self.null_gold:
                continue
            s += f" y={j} " + tab
        print(s)

        # Print data
        for i in range(m):
            # Skip null predictions row if necessary
            if i == 0 and not self.null_pred:
                continue
            s = margin + f" l={i} " + tab
            for j in range(n):
                # Skip null gold if necessary
                if j == 0 and not self.null_gold:
                    continue
                else:
                    if i == j and mark_diag and normalize:
                        s = s[:-1] + "*"
                    if normalize:
                        s += f"{mat[i,j]/sum(mat[i,1:]):>5.3f}" + tab
                    else:
                        s += f"{mat[i,j]:^5d}" + tab
            print(s)
