from typing import Callable, List, NamedTuple

import numpy as np
import sklearn.metrics as skmetrics

from .utils import filter_labels


class Metric(NamedTuple):
    """Specifies a metric function and the subset of [golds, preds, probs] it expects"""

    func: Callable[..., float]
    inputs: List[str] = ["golds", "preds"]


def metric_score(
    golds: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    metric: str,
    ignore_in_golds: List[int] = [],
    ignore_in_preds: List[int] = [],
    **kwargs,
) -> float:
    """A method for evaluating a standard metric on a set of predictions/probabilities

    Parameters
    ----------
    golds
        An array of gold (int) labels
    preds
        An array of (int) predictions
    probs
        An [n_datapoints, n_classes] array of probabilistic predictions
    metric
        The name of the metric to calculate
    ignore_in_golds
        A list of labels in golds whose corresponding examples should be ignored
    ignore_in_preds
        A list of labels in predictions whose corresponding examples should be ignored

    Returns
    -------
    float
        The value of the requested metric

    Raises
    ------
    ValueError
        The requested metric is not currently supported
    ValueError
        The user attempted to calculate roc_auc score for a non-binary problem
    """
    if metric not in METRICS:
        msg = f"The metric you provided ({metric}) is not currently implemented."
        raise ValueError(msg)

    # Optionally filter out examples (e.g., abstain predictions or unknown labels)
    golds, preds, probs = filter_labels(
        golds, preds, probs, ignore_in_golds, ignore_in_preds
    )

    # Pass the metric function its requested args
    func, input_names = METRICS[metric]
    input_map = {"golds": golds, "preds": preds, "probs": probs}
    inputs = [input_map[input] for input in input_names]

    return func(*inputs, **kwargs)


def _coverage_score(preds: np.ndarray) -> float:
    """A helper used by metric_score() to calculate coverage (percent not abstained)"""
    return np.sum(preds != 0) / len(preds)


def _roc_auc_score(golds: np.ndarray, probs: np.ndarray) -> float:
    """A helper used by metric_score() to calculate roc_auc score (see sklearn)"""
    if not probs.shape[1] == 2:
        raise ValueError(
            "Metric roc_auc is currently only defined for binary problems."
        )
    return skmetrics.roc_auc_score(golds, probs[:, 0])


# See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# for details on the definitions and available kwargs for all metrics from scikit-learn
METRICS = {
    "accuracy": Metric(skmetrics.accuracy_score),
    "coverage": Metric(_coverage_score, ["preds"]),
    "precision": Metric(skmetrics.precision_score),
    "recall": Metric(skmetrics.recall_score),
    "f1": Metric(skmetrics.f1_score),
    "fbeta": Metric(skmetrics.fbeta_score),
    "matthews_corrcoef": Metric(skmetrics.matthews_corrcoef),
    "roc_auc": Metric(_roc_auc_score, ["golds", "probs"]),
}
