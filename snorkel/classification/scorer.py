from functools import partial
from typing import Callable, Dict, List, Mapping, Optional

from snorkel.analysis.metrics import METRICS, metric_score
from snorkel.types import ArrayLike


class Scorer:
    """Calculate one or more scores from user-specified and/or user-defined metrics.

    Parameters
    ----------
    metrics
        A list of metric names, all of which are defined in METRICS
    custom_metric_funcs:
        An optional dictionary mapping the names of custom metrics to the functions
        that produce them. Each custom metric function should accept golds, preds, and
        probs as input (just like the standard metrics in METRICS) and return either a
        single score (float) or a dictionary of metric names to scores (if the function
        calculates multiple values, for example). See the unit tests for an example.

    Attributes
    ----------
    metrics
        A dictionary mapping metric names to the corresponding functions for calculating
        that metric

    Raises
    ------
    ValueError
        If a specified standard metric is not found in the METRICS dictionary
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        custom_metric_funcs: Optional[Mapping[str, Callable[..., float]]] = None,
    ) -> None:

        self.metrics: Dict[str, Callable[..., float]]
        if metrics:
            for metric in metrics:
                if metric not in METRICS:
                    raise ValueError(f"Unrecognized metric: {metric}")
            self.metrics = {m: partial(metric_score, metric=m) for m in metrics}
        else:
            self.metrics = {}

        if custom_metric_funcs is not None:
            self.metrics.update(custom_metric_funcs)

    def score(
        self, golds: ArrayLike, preds: ArrayLike, probs: ArrayLike
    ) -> Dict[str, float]:
        """Calculate one or more scores from user-specified and/or user-defined metrics.

        Parameters
        ----------
        golds
            Gold (aka ground truth) labels (integers)
        preds
            Predictions (integers)
        probs:
            Probabilities (floats)

        Returns
        -------
        Dict[str, float]
            A dictionary mapping metric names to metric scores

        Raises
        ------
        ValueError
            If no gold labels were provided
        """
        if len(golds) == 0:  # type: ignore
            raise ValueError("Cannot score empty labels")

        metric_dict = dict()

        for metric_name, metric in self.metrics.items():

            score = metric(golds, preds, probs)

            if isinstance(score, dict):
                metric_dict.update(score)
            else:
                metric_dict[metric_name] = score

        return metric_dict
