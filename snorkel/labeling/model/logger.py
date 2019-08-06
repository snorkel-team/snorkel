import logging
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional


class Logger:
    """Class for logging LabelModel.

    Parameters
    ----------
    log_freq
        Number of units at which to log model

    Attributes
    ----------
    log_freq
        Number of units at which to log model
    unit_count
        Running total of number of units passed without logging
    """

    def __init__(self, log_freq: int) -> None:
        self.log_freq = log_freq
        self.unit_count = -1

    def check(self) -> bool:
        """Check if the logging frequency has been met.

        Returns
        -------
        bool
            Whether to log or not based on logging frequency
        """
        self.unit_count += 1
        return self.unit_count % self.log_freq == 0

    def log(self, metrics_dict: Dict[str, float]) -> None:
        """Print all metrics in metrics_dict to screen.

        Parameters
        ----------
        metrics_dict
            Dictionary of metric names (keys) and values to log

        Raises
        ------
        Exception
            If metric names formatted incorrectly
        """
        score_strings: DefaultDict[str, List[str]] = defaultdict(list)
        for full_name, value in metrics_dict.items():
            task: Optional[str]
            if full_name.count("/") == 2:
                task, split, metric = full_name.split("/")
            elif full_name.count("/") == 1:
                task = None
                split, metric = full_name.split("/")
            else:
                msg = f"Metric should have form task/split/metric or split/metric, not: {full_name}"
                raise Exception(msg)

            if task:
                metric_name = f"{task}/{metric}"
            else:
                metric_name = metric
            if isinstance(value, float):
                score_strings[split].append(f"{metric_name}={value:0.3f}")
            else:
                score_strings[split].append(f"{metric_name}={value}")

        string = f"[{self.unit_count} epochs]:"

        if score_strings["train"]:
            train_scores = f"{', '.join(score_strings['train'])}"
            string += f" TRAIN:[{train_scores}]"
        if score_strings["valid"]:
            valid_scores = f"{', '.join(score_strings['valid'])}"
            string += f" VALID:[{valid_scores}]"
        logging.info(string)
