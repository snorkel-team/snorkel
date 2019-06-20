import logging
import os
from datetime import datetime
from typing import Dict

import torch

from snorkel.mtl.utils import recursive_merge_dicts

from .checkpointer import Checkpointer
from .log_writer import LogWriter
from .tensorboard_writer import TensorBoardWriter

logger_default_config = {
    "log_dir": "logs",  # The path to the directory under which logs will be written
    "counter_unit": "batches",  # [points, batches, epochs]
    "evaluation_freq": 2,  # Evaluate performance every this many counter_units
    "writer_config": {"writer": None, "verbose": True},  # [json, tensorboard]
    "checkpointing": False,
    "checkpointer_config": {
        "checkpoint_dir": None,
        "checkpoint_factor": 1,  # Checkpoint every this many evaluations
        "checkpoint_metric": "model/train/all/loss:min",
        "checkpoint_task_metrics": None,  # task_metric_name:mode
        "checkpoint_runway": 0,  # checkpointing runway (no checkpointing before k unit)
        "checkpoint_clear": True,  # whether to clear intermediate checkpoints
    },
}


class LogManager(object):
    """A class to manage logging during training progress

    :param n_batches_per_epoch: total number batches per epoch
    :type n_batches_per_epoch: int
    :param verbose: print out the log or not
    :type verbose: bool
    """

    def __init__(self, n_batches_per_epoch: int, **kwargs) -> None:

        self.config = recursive_merge_dicts(logger_default_config, kwargs)
        self.n_batches_per_epoch = n_batches_per_epoch

        # Create log directory for this run
        date = datetime.now().strftime("%Y_%m_%d")
        time = datetime.now().strftime("%H_%M_%S")
        self.log_path = os.path.join(self.config["log_dir"], date, time)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # Set up counter unit
        self.counter_unit = self.config["counter_unit"]
        if self.counter_unit not in ["points", "batches", "epochs"]:
            raise ValueError(f"Unrecognized counter_unit: {self.counter_unit}")

        # Set up evaluation frequency
        self.evaluation_freq = self.config["evaluation_freq"]
        logging.info(f"Evaluating every {self.evaluation_freq} {self.counter_unit}.")

        # Set up writer and checkpointer
        self._init_writer()
        self._init_checkpointer()

        # Set up number of X passed since last evaluation/checkpointing and total
        self.point_count = 0
        self.point_total = 0

        self.batch_count = 0
        self.batch_total = 0

        self.epoch_count = 0.0
        self.epoch_total = 0.0

        self.unit_count = 0.0
        self.unit_total = 0.0

        # Set up count that triggers the evaluation since last checkpointing
        self.trigger_count = 0

    def update(self, batch_size: int) -> None:
        """Update the count and total number"""

        # Update number of points
        self.point_count += batch_size
        self.point_total += batch_size

        # Update number of batches
        self.batch_count += 1
        self.batch_total += 1

        # Update number of epochs
        self.epoch_count = self.batch_count / self.n_batches_per_epoch
        self.epoch_total = self.batch_total / self.n_batches_per_epoch

        # Update number of units
        if self.counter_unit == "points":
            self.unit_count = self.point_count
            self.unit_total = self.point_total
        if self.counter_unit == "batches":
            self.unit_count = self.batch_count
            self.unit_total = self.batch_total
        elif self.counter_unit == "epochs":
            self.unit_count = self.epoch_count
            self.unit_total = self.epoch_total

    def trigger_evaluation(self) -> bool:
        """Check if triggers the evaluation"""
        satisfied = self.unit_count >= self.evaluation_freq
        if satisfied:
            self.trigger_count += 1
            self.reset()
        return satisfied

    def trigger_checkpointing(self) -> bool:
        """Check if triggers the checkpointing"""
        if not self.checkpointing:
            return False
        satisfied = self.trigger_count >= self.checkpointing_freq
        if satisfied:
            self.trigger_count = 0
        return satisfied

    def reset(self) -> None:
        """Reset the counter"""
        self.point_count = 0
        self.batch_count = 0
        self.epoch_count = 0
        self.unit_count = 0

    def write_log(self, metric_dict: Dict[str, float]) -> None:
        if self.writer:
            for metric_name, metric_value in metric_dict.items():
                self.writer.add_scalar(metric_name, metric_value, self.batch_total)

    def checkpoint_model(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        metric_dict: Dict[str, str],
    ) -> None:
        self.checkpointer.checkpoint(
            self.unit_total, model, optimizer, lr_scheduler, metric_dict
        )

    def close(self, model):
        if self.writer:
            self.writer.close()
        if self.checkpointing:
            self.checkpointer.clear()
            model = self.checkpointer.load_best_model(model)
        return model

    def _init_checkpointer(self) -> None:
        if self.config["checkpointing"]:
            self.checkpointing = True

            # Set up checkpointing frequency
            self.checkpointing_freq = int(
                self.config["checkpointer_config"]["checkpoint_factor"]
            )
            logging.info(
                f"Checkpointing every "
                f"{self.checkpointing_freq * self.evaluation_freq} {self.counter_unit}."
            )

            if not self.config["checkpointer_config"]["checkpoint_dir"]:
                self.config["checkpointer_config"]["checkpoint_dir"] = os.path.join(
                    self.log_path, "checkpoints"
                )

            # Set up checkpointer
            self.checkpointer = Checkpointer(self.config)
        else:
            self.checkpointing = False
            logging.info("No checkpointing.")

    def _init_writer(self) -> None:
        writer_opt = self.config["writer_config"]["writer"]

        if writer_opt is None:
            self.writer = None
        elif writer_opt == "json":
            self.writer = LogWriter(self.log_path)
        elif writer_opt == "tensorboard":
            self.writer = TensorBoardWriter(self.log_path)
        else:
            raise ValueError(f"Unrecognized writer option '{writer_opt}'")
