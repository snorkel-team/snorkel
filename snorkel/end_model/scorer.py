import logging

from snorkel.analysis.metrics import METRICS

logger = logging.getLogger(__name__)


class Scorer(object):
    """A class to score tasks

    :param metrics: a list of metric names which provides in emmental (e.g., accuracy)
    :type metrics: list
    :param custom_metric_funcs: a dict of custom metrics where key is the metric
    name and value is the metric function which takes gold, pred, prob as input
    :type custom_metric_funcs: dict
    """

    def __init__(self, metrics=[], custom_metric_funcs={}):
        self.metrics = dict()
        for metric in metrics:
            if metric not in METRICS:
                raise ValueError(f"Unrecognized metric: {metric}")
            self.metrics[metric] = METRICS[metric]

        self.metrics.update(custom_metric_funcs)

    def score(self, gold, pred, prob):
        metric_dict = dict()

        for metric_name, metric in self.metrics.items():
            # Handle no examples
            if len(gold) == 0:
                metric_dict[metric_name] = float("nan")
                continue

            score = metric(gold, pred, prob)

            if isinstance(score, dict):
                metric_dict.update(score)
            else:
                metric_dict[metric_name] = score

        return metric_dict
