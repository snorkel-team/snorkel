import numpy as np

from .utils import arraylike_to_numpy


def accuracy_score(golds, preds, probs=None, ignore_in_gold=[], ignore_in_pred=[]):
    """
    Calculate (micro) accuracy.
    Args:
        golds: A 1d array-like of golds labels
        preds: A 1d array-like of predicted labels (assuming abstain = 0)
        probs: Unused (kept for compatibility with expected metric func signature)
        ignore_in_gold: A list of labels for which elements having that golds
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that preds
            label will be ignored.

    Returns:
        A float, the (micro) accuracy score
    """
    golds, preds, probs = _preprocess(golds, preds, probs, ignore_in_gold, ignore_in_pred)

    if len(golds) and len(preds):
        acc = np.sum(golds == preds) / len(golds)
    else:
        acc = 0

    return acc


def coverage_score(golds, preds, probs=None, ignore_in_gold=[], ignore_in_pred=[]):
    """
    Calculate (global) coverage.
    Args:
        golds: A 1d array-like of golds labels
        preds: A 1d array-like of predicted labels (assuming abstain = 0)
        probs: Unused (kept for compatibility with expected metric func signature)
        ignore_in_gold: A list of labels for which elements having that golds
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that preds
            label will be ignored.

    Returns:
        A float, the (global) coverage score
    """
    golds, preds, probs = _preprocess(golds, preds, probs, ignore_in_gold, ignore_in_pred)

    return np.sum(preds != 0) / len(preds)


def precision_score(
    golds, preds, probs=None, pos_label=1, ignore_in_gold=[], ignore_in_pred=[]
):
    """
    Calculate precision for a single class.
    Args:
        golds: A 1d array-like of golds labels
        preds: A 1d array-like of predicted labels (assuming abstain = 0)
        probs: Unused (kept for compatibility with expected metric func signature)
        ignore_in_gold: A list of labels for which elements having that golds
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that preds
            label will be ignored.
        pos_label: The class label to treat as positive for precision

    Returns:
        pre: The (float) precision score
    """
    golds, preds, probs = _preprocess(golds, preds, probs, ignore_in_gold, ignore_in_pred)

    positives = np.where(preds == pos_label, 1, 0).astype(bool)
    trues = np.where(golds == pos_label, 1, 0).astype(bool)
    TP = np.sum(positives * trues)
    FP = np.sum(positives * np.logical_not(trues))

    if TP or FP:
        pre = TP / (TP + FP)
    else:
        pre = 0

    return pre


def recall_score(
    golds, preds, probs=None, pos_label=1, ignore_in_gold=[], ignore_in_pred=[]
):
    """
    Calculate recall for a single class.
    Args:
        golds: A 1d array-like of golds labels
        preds: A 1d array-like of predicted labels (assuming abstain = 0)
        probs: Unused (kept for compatibility with expected metric func signature)
        ignore_in_gold: A list of labels for which elements having that golds
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that preds
            label will be ignored.
        pos_label: The class label to treat as positive for recall

    Returns:
        rec: The (float) recall score
    """
    golds, preds, probs = _preprocess(golds, preds, probs, ignore_in_gold, ignore_in_pred)

    positives = np.where(preds == pos_label, 1, 0).astype(bool)
    trues = np.where(golds == pos_label, 1, 0).astype(bool)
    TP = np.sum(positives * trues)
    FN = np.sum(np.logical_not(positives) * trues)

    if TP or FN:
        rec = TP / (TP + FN)
    else:
        rec = 0

    return rec


def fbeta_score(
    golds, preds, probs=None, pos_label=1, beta=1.0, ignore_in_gold=[], ignore_in_pred=[]
):
    """
    Calculate recall for a single class.
    Args:
        golds: A 1d array-like of golds labels
        preds: A 1d array-like of predicted labels (assuming abstain = 0)
        probs: Unused (kept for compatibility with expected metric func signature)
        ignore_in_gold: A list of labels for which elements having that golds
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that preds
            label will be ignored.
        pos_label: The class label to treat as positive for f-beta
        beta: The beta to use in the f-beta score calculation

    Returns:
        fbeta: The (float) f-beta score
    """
    golds, preds, probs = _preprocess(golds, preds, probs, ignore_in_gold, ignore_in_pred)
    pre = precision_score(golds, preds, pos_label=pos_label)
    rec = recall_score(golds, preds, pos_label=pos_label)

    if pre or rec:
        fbeta = (1 + beta ** 2) * (pre * rec) / ((beta ** 2 * pre) + rec)
    else:
        fbeta = 0

    return fbeta


def f1_score(golds, preds, probs=None, **kwargs):
    return fbeta_score(golds, preds, probs, beta=1.0, **kwargs)


def _get_mask(golds, preds, ignore_in_gold, ignore_in_pred):
    """Remove from golds, preds all items with labels designated to ignore."""
    mask = np.ones_like(golds).astype(bool)
    for x in ignore_in_gold:
        mask *= np.where(golds != x, 1, 0).astype(bool)
    for x in ignore_in_pred:
        mask *= np.where(preds != x, 1, 0).astype(bool)

    return mask


def _preprocess(golds, preds, probs, ignore_in_gold, ignore_in_pred):
    golds = arraylike_to_numpy(golds) if golds is not None else None
    preds = arraylike_to_numpy(preds) if preds is not None else None
    if ignore_in_gold or ignore_in_pred:
        mask = _get_mask(golds, preds, ignore_in_gold, ignore_in_pred)
        golds = golds[mask] if golds is not None else None
        preds = preds[mask] if preds is not None else None
        probs = probs[mask] if probs is not None else None
    return golds, preds, probs


METRICS = {
    "accuracy": accuracy_score,
    "coverage": coverage_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "fbeta": fbeta_score,
}


def metric_score(golds, preds, probs, metric, probs=None, **kwargs):
    if metric not in METRICS:
        msg = f"The metric you provided ({metric}) is not supported."
        raise ValueError(msg)
    else:
        return METRICS[metric](golds, preds, probs, **kwargs)
