import glob
import logging
import os
from shutil import copyfile
from typing import Any, Dict, Iterable, List, Optional, Set

from snorkel.classification.multitask_classifier import MultitaskClassifier
from snorkel.types import Config

Metrics = Dict[str, float]


class CheckpointerConfig(Config):
    """Manager for checkpointing model.

    Parameters
    ----------
    checkpoint_dir
        The path to a directory where checkpoints will be saved
        The Trainer will set this to the log directory if it is None
    checkpoint_factor
        Check for a best model every this many evaluations. For example, if
        evaluation_freq is 0.5 epochs and checkpoint_factor is 2, then checkpointing
        will be attempted every 1 epochs.
    checkpoint_metric
        The metric to checkpoint on, of the form "task/dataset/split/metric:mode" where
        mode is "min" or "max".
    checkpoint_task_metrics
        Additional metrics to save best models for. Note that the best model according
        to `checkpoint_metric` will be the one that is loaded after training and used
        for early stopping.
    checkpoint_runway
        No checkpointing will occur for the first this many checkpoint_units
    checkpoint_clear
        If True, clear all checkpoints besides the best so far.
    """

    checkpoint_dir: str = "checkpoints"
    checkpoint_factor: int = 1
    checkpoint_metric: str = "model/all/train/loss:min"
    checkpoint_task_metrics: Optional[List[str]] = None
    checkpoint_runway: int = 0
    checkpoint_clear: bool = True


class Checkpointer:
    """Manager for checkpointing model.

    Parameters
    ----------
    counter_unit
        The unit to use when determining when its time to checkpoint (one of
        ["epochs", "batches", "points"]); must match the counter_unit of LogManager
    evaluation_freq
        How frequently the model is being evaluated (this is the maximum frequency that
        checkpointing can occur, which will happen if checkpoint_factor==1)
    kwargs
        Config merged with ``default_config["checkpointer_config"]``
    """

    def __init__(
        self, counter_unit: str, evaluation_freq: float, **kwargs: Any
    ) -> None:
        self.config = CheckpointerConfig(**kwargs)
        self._validate_config()

        # Pull out checkpoint settings
        self.checkpoint_unit = counter_unit
        self.checkpoint_dir = self.config.checkpoint_dir
        self.checkpoint_clear = self.config.checkpoint_clear
        self.checkpoint_runway = self.config.checkpoint_runway
        self.checkpoint_factor = self.config.checkpoint_factor
        self.checkpoint_condition_met = False

        # Collect all metrics to checkpoint
        self.checkpoint_metric = self._make_metric_map([self.config.checkpoint_metric])
        self.checkpoint_task_metrics = self._make_metric_map(
            self.config.checkpoint_task_metrics
        )
        self.checkpoint_task_metrics.update(self.checkpoint_metric)

        # Create checkpoint directory if necessary
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Set checkpoint frequency
        self.checkpoint_freq = evaluation_freq * self.checkpoint_factor
        if self.checkpoint_freq <= 0:
            raise ValueError(
                f"Invalid checkpoint freq {self.checkpoint_freq}, "
                f"must be greater than 0."
            )

        logging.info(
            f"Save checkpoints at '{self.checkpoint_dir}' every "
            f"{self.checkpoint_freq} {self.checkpoint_unit}."
        )

        if self.checkpoint_runway > 0:
            logging.info(
                f"No checkpoints will be saved before {self.checkpoint_runway} "
                f"{self.checkpoint_unit}."
            )

        self.best_metric_dict: Dict[str, float] = {}

    def checkpoint(
        self, iteration: float, model: MultitaskClassifier, metric_dict: Metrics
    ) -> None:
        """Check if iteration and current metrics necessitate a checkpoint.

        Parameters
        ----------
        iteration
            Current training iteration
        model
            Model to checkpoint
        metric_dict
            Current performance metrics for model
        """
        # Check if the checkpoint_runway condition is met
        if iteration < self.checkpoint_runway:
            return
        elif not self.checkpoint_condition_met and iteration >= self.checkpoint_runway:
            self.checkpoint_condition_met = True
            logging.info(
                "checkpoint_runway condition has been met. Start checkpointing."
            )

        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_{iteration}.pth"
        model.save(checkpoint_path)
        logging.info(
            f"Save checkpoint at {iteration} {self.checkpoint_unit} "
            f"at {checkpoint_path}."
        )

        if not set(self.checkpoint_task_metrics.keys()).isdisjoint(
            set(metric_dict.keys())
        ):
            new_best_metrics = self._is_new_best(metric_dict)
            for metric in new_best_metrics:
                copyfile(
                    checkpoint_path,
                    f"{self.checkpoint_dir}/best_model_"
                    f"{metric.replace('/', '_')}.pth",
                )

                logging.info(
                    f"Save best model of metric {metric} at {self.checkpoint_dir}"
                    f"/best_model_{metric.replace('/', '_')}.pth"
                )

    def _is_new_best(self, metric_dict: Metrics) -> Set[str]:

        best_metric = set()

        for metric in metric_dict:
            if metric not in self.checkpoint_task_metrics:
                continue
            if metric not in self.best_metric_dict:
                self.best_metric_dict[metric] = metric_dict[metric]
                best_metric.add(metric)
            elif (
                self.checkpoint_task_metrics[metric] == "max"
                and metric_dict[metric] > self.best_metric_dict[metric]
            ):
                self.best_metric_dict[metric] = metric_dict[metric]
                best_metric.add(metric)
            elif (
                self.checkpoint_task_metrics[metric] == "min"
                and metric_dict[metric] < self.best_metric_dict[metric]
            ):
                self.best_metric_dict[metric] = metric_dict[metric]
                best_metric.add(metric)

        return best_metric

    def clear(self) -> None:
        """Clear existing checkpoint files, besides the best-yet model."""
        if self.checkpoint_clear:
            logging.info("Clear all checkpoints other than best so far.")
            file_list = glob.glob(f"{self.checkpoint_dir}/checkpoint_*.pth")
            for fname in file_list:
                os.remove(fname)

    def load_best_model(self, model: MultitaskClassifier) -> MultitaskClassifier:
        """Load the best model from the checkpoint."""
        metric = list(self.checkpoint_metric.keys())[0]
        if metric not in self.best_metric_dict:  # pragma: no cover
            logging.info("No best model found, use the original model.")
        else:
            # Load the best model of checkpoint_metric
            best_model_path = (
                f"{self.checkpoint_dir}/best_model_{metric.replace('/', '_')}.pth"
            )
            logging.info(f"Loading the best model from {best_model_path}.")
            model.load(best_model_path)

        return model

    def _validate_config(self) -> None:
        split_checkpoint_metric = self.config.checkpoint_metric.split("/")
        if len(split_checkpoint_metric) != 4:
            raise ValueError(
                "checkpoint_metric must be formatted 'task/dataset/split/metric:mode'."
            )

        if self.config.checkpoint_runway < 0:
            raise ValueError(
                f"Invalid checkpoint_runway {self.config.checkpoint_runway}, "
                f"must be greater than or equal to 0."
            )

    def _make_metric_map(
        self, metric_mode_iter: Optional[Iterable[str]]
    ) -> Dict[str, str]:
        if metric_mode_iter is None:
            return {}

        metric_mode_map = dict()
        for metric_mode in metric_mode_iter:
            try:
                metric, mode = metric_mode.split(":")
            except ValueError:
                raise ValueError(
                    f"Metric must be of the form 'metric_name:mode' where mode is "
                    f"'max' or 'min'. Instead, got {metric_mode}"
                )
            if mode not in ["min", "max"]:
                raise ValueError(
                    f"Unrecognized checkpoint metric mode {mode} for metric {metric}, "
                    f"must be 'min' or 'max'."
                )
            metric_mode_map[metric] = mode

        return metric_mode_map
