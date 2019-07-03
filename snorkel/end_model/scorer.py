from functools import partial
from typing import Callable, List, Mapping, Optional

from snorkel.analysis.metrics import METRICS, metric_score


class Scorer(object):
    """A class to score tasks

    :param metrics: a list of metric names which provides in emmental (e.g., accuracy)
    :type metrics: list
    :param custom_metric_funcs: a dict of custom metrics where key is the metric
    name and value is the metric function which takes golds, preds, probs as input
    and returns either a dict of metric names and scores or a single score
    :type custom_metric_funcs: dict
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        custom_metric_funcs: Optional[Mapping[str, Callable[..., float]]] = None,
    ):

        if metrics:
            for metric in metrics:
                if metric not in METRICS:
                    raise ValueError(f"Unrecognized metric: {metric}")
            self.metrics = {m: partial(metric_score, metric=m) for m in metrics}
        else:
            self.metrics = {}

        if custom_metric_funcs is not None:
            self.metrics.update(custom_metric_funcs)

    def score(self, golds, preds, probs):
        metric_dict = dict()

        for metric_name, metric in self.metrics.items():
            # Handle no examples
            if len(golds) == 0:
                metric_dict[metric_name] = float("nan")
                continue

            if metric_name in self.metrics:
                score = metric(golds, preds, probs)

            if isinstance(score, dict):
                metric_dict.update(score)
            else:
                metric_dict[metric_name] = score

        return metric_dict
