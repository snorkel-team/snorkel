import glob
import logging
import os
from shutil import copyfile
from typing import Dict, List

from snorkel.classification.snorkel_config import default_config
from snorkel.classification.utils import recursive_merge_dicts


class Checkpointer(object):
    """Checkpointing Training logging class to log train infomation"""

    def __init__(self, **kwargs):

        # Checkpointer requires both checkpointer_config and log_manager_config
        # Use recursive_merge_dict instead of dict.update() to ensure copies are made
        checkpointer_config = recursive_merge_dicts(
            default_config["checkpointer_config"],
            default_config["log_manager_config"],
            misses="insert",
        )
        self.config = recursive_merge_dicts(checkpointer_config, kwargs)

        # Pull out checkpoint settings
        self.checkpoint_dir = self.config["checkpoint_dir"]
        self.checkpoint_unit = self.config["counter_unit"]
        self.checkpoint_clear = self.config["checkpoint_clear"]
        self.checkpoint_runway = self.config["checkpoint_runway"]
        self.checkpoint_factor = self.config["checkpoint_factor"]
        self.checkpoint_condition_met = False

        if self.checkpoint_dir is None:
            raise ValueError("Checkpointing is on but no checkpoint_dir was specified.")

        # Collect all metrics to checkpoint
        self.checkpoint_metric = self._make_metric_map(
            [self.config["checkpoint_metric"]]
        )
        self.checkpoint_task_metrics = self._make_metric_map(
            self.config["checkpoint_task_metrics"]
        )
        self.checkpoint_task_metrics.update(self.checkpoint_metric)

        # Create checkpoint directory if necessary
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Set checkpoint frequency
        self.checkpoint_freq = self.config["evaluation_freq"] * self.checkpoint_factor
        if self.checkpoint_freq <= 0:
            raise ValueError(
                f"Invalid checkpoint freq {self.checkpoint_freq}, "
                f"must be greater 0."
            )

        logging.info(
            f"Save checkpoints at {self.checkpoint_dir} every "
            f"{self.checkpoint_freq} {self.checkpoint_unit}."
        )

        logging.info(
            f"No checkpoints will be saved before {self.checkpoint_runway} "
            f"{self.checkpoint_unit}."
        )

        self.best_metric_dict: Dict[str, float] = {}

    def checkpoint(self, iteration, model, optimizer, lr_scheduler, metric_dict):
        # Check if the checkpoint_runway condition is met
        if iteration < self.checkpoint_runway:
            return
        elif not self.checkpoint_condition_met and iteration >= self.checkpoint_runway:
            self.checkpoint_condition_met = True
            logging.info(
                f"checkpoint_runway condition has been met. Start checkpointing."
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
            new_best_metrics = self.is_new_best(metric_dict)
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

    def is_new_best(self, metric_dict):
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

    def clear(self):
        if self.checkpoint_clear:
            logging.info("Clear all immediate checkpoints.")
            file_list = glob.glob(f"{self.checkpoint_dir}/checkpoint_*.pth")
            for file in file_list:
                os.remove(file)

    def load_best_model(self, model):
        """Load the best model from the checkpoint."""
        if list(self.checkpoint_metric.keys())[0] not in self.best_metric_dict:
            logging.info(f"No best model found, use the original model.")
        else:
            # Load the best model of checkpoint_metric
            metric = list(self.checkpoint_metric.keys())[0]
            best_model_path = (
                f"{self.checkpoint_dir}/best_model_{metric.replace('/', '_')}.pth"
            )
            logging.info(f"Loading the best model from {best_model_path}.")
            model.load(best_model_path)

        return model

    def _make_metric_map(self, metric_mode_list: List[str]):
        if metric_mode_list is None:
            return {}

        metric_mode_map = dict()
        for metric_mode in metric_mode_list:
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
