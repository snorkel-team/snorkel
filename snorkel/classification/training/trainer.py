import logging
from collections import defaultdict
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm

from snorkel.classification.snorkel_classifier import SnorkelClassifier
from snorkel.classification.snorkel_config import default_config
from snorkel.classification.training import (
    Checkpointer,
    LogManager,
    LogWriter,
    TensorBoardWriter,
)
from snorkel.classification.training.schedulers import batch_schedulers
from snorkel.classification.utils import recursive_merge_dicts


class Trainer(object):
    """A class for multi-task learning.

    :param config: The learning config
    :type config: dict
    """

    def __init__(self, name=None, **kwargs):
        self.config = recursive_merge_dicts(default_config, kwargs, misses="insert")
        self.name = name if name is not None else type(self).__name__

    def train_model(self, model: SnorkelClassifier, dataloaders):
        """The learning procedure of MTL

        :param model: The multi-task model that needs to learn
        :type model: SnorkelClassifier
        :param dataloaders: a list of dataloaders used to learn the model
        :type dataloaders: list
        """
        self._check_dataloaders(dataloaders)

        # Identify the dataloaders to train on
        train_dataloaders = [
            dl for dl in dataloaders if dl.dataset.split == self.config["train_split"]
        ]

        # Calculate the total number of batches per epoch
        self.n_batches_per_epoch = sum(
            [len(dataloader) for dataloader in train_dataloaders]
        )

        # Set training helpers
        self._set_log_writer()
        self._set_checkpointer()
        self._set_log_manager()
        self._set_optimizer(model)
        self._set_lr_scheduler(model)
        self._set_batch_scheduler()

        # Set to training mode
        model.train()

        logging.info(f"Start training...")

        self.metrics: Dict[str, float] = dict()
        self._reset_losses()

        for epoch_num in range(self.config["n_epochs"]):
            batches = tqdm(
                enumerate(self.batch_scheduler.get_batches(train_dataloaders)),
                total=self.n_batches_per_epoch,
                disable=(not self.config["progress_bar"]),
                desc=f"Epoch {epoch_num}:",
            )
            for batch_num, (batch, dataloader) in batches:
                X_dict, Y_dict = batch

                total_batch_num = epoch_num * self.n_batches_per_epoch + batch_num
                batch_size = len(next(iter(Y_dict.values())))

                # Update lr using lr scheduler
                self._update_lr_scheduler(model, total_batch_num)

                # Set gradients of all model parameters to zero
                self.optimizer.zero_grad()

                # Perform forward pass and calcualte the loss and count
                loss_dict, count_dict = model.calculate_loss(X_dict, Y_dict)

                # Update running loss and count
                for task_name in loss_dict.keys():
                    identifier = "/".join(
                        [
                            task_name,
                            dataloader.dataset.name,
                            dataloader.dataset.split,
                            "loss",
                        ]
                    )
                    self.running_losses[identifier] += (
                        loss_dict[task_name].item() * count_dict[task_name]
                    )
                    self.running_counts[task_name] += count_dict[task_name]

                # Skip the backward pass if no loss is calcuated
                if not loss_dict:
                    continue

                # Calculate the average loss
                loss = torch.stack(list(loss_dict.values())).sum()

                # Perform backward pass to calculate gradients
                loss.backward()

                # Clip gradient norm
                if self.config["optimizer_config"]["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config["optimizer_config"]["grad_clip"]
                    )

                # Update the parameters
                self.optimizer.step()

                # Update metrics
                self.metrics.update(self._logging(model, dataloaders, batch_size))

                batches.set_postfix(self.metrics)

        model = self.log_manager.close(model)

    def _check_dataloaders(self, dataloaders):
        """ Validates dataloaders given training config"""
        if not isinstance(dataloaders, list):
            raise Exception("Trainer.train_model() expects a list of DictDataLoaders.")

        train_split = self.config["train_split"]
        valid_split = self.config["valid_split"]
        test_split = self.config["test_split"]

        all_splits = [train_split, valid_split, test_split]
        if not all([dl.dataset.split in all_splits for dl in dataloaders]):
            raise ValueError(f"Dataloader splits must be one of {all_splits}")

        if not any([dl.dataset.split == train_split for dl in dataloaders]):
            raise ValueError(
                f"Cannot find any dataloaders with split matching train split: "
                f'{self.config["train_split"]}.'
            )

    def _set_log_writer(self):
        if self.config["logging"]:
            config = self.config["log_writer_config"]
            if config["writer"] == "json":
                self.log_writer = LogWriter(**config)
            elif config["writer"] == "tensorboard":
                self.log_writer = TensorBoardWriter(**config)
            else:
                raise ValueError(f"Unrecognized writer option: {config['writer']}")
        else:
            self.log_writer = None

    def _set_checkpointer(self):
        if self.config["checkpointing"]:
            checkpointer_config = self.config["checkpointer_config"]
            log_manager_config = self.config["log_manager_config"]

            # Default checkpoint_dir to log_dir if available
            if checkpointer_config["checkpoint_dir"] is None and self.config["logging"]:
                checkpointer_config["checkpoint_dir"] = self.log_writer.log_dir

            self.checkpointer = Checkpointer(
                **checkpointer_config, **log_manager_config
            )
        else:
            self.checkpointer = None

    def _set_log_manager(self):
        """Set logging manager."""
        self.log_manager = LogManager(
            self.n_batches_per_epoch,
            log_writer=self.log_writer,
            checkpointer=self.checkpointer,
            **self.config["log_manager_config"],
        )

    def _set_optimizer(self, model):
        """Set optimizer for learning process."""

        # TODO: add more optimizer support and fp16
        optimizer_config = self.config["optimizer_config"]
        opt = optimizer_config["optimizer"]

        parameters = filter(lambda p: p.requires_grad, model.parameters())

        if opt == "sgd":
            optimizer = optim.SGD(
                parameters,
                lr=optimizer_config["lr"],
                **optimizer_config["sgd_config"],
                weight_decay=optimizer_config["l2"],
            )
        elif opt == "adam":
            optimizer = optim.Adam(
                parameters,
                lr=optimizer_config["lr"],
                **optimizer_config["adam_config"],
                weight_decay=optimizer_config["l2"],
            )
        elif opt == "adamax":
            optimizer = optim.Adamax(
                parameters,
                lr=optimizer_config["lr"],
                **optimizer_config["adamax_config"],
                weight_decay=optimizer_config["l2"],
            )
        else:
            raise ValueError(f"Unrecognized optimizer option '{opt}'")

        logging.info(f"Using optimizer {optimizer}")

        self.optimizer = optimizer

    def _set_lr_scheduler(self, model):
        """Set learning rate scheduler for learning process."""

        # Set warmup scheduler
        self._set_warmup_scheduler(model)

        # Set lr scheduler
        # TODO: add more lr scheduler support
        opt = self.config["lr_scheduler_config"]["lr_scheduler"]
        lr_scheduler_config = self.config["lr_scheduler_config"]

        if opt == "constant":
            lr_scheduler = None
        elif opt == "linear":
            total_steps = self.n_batches_per_epoch * self.config["n_epochs"]
            linear_decay_func = lambda x: (total_steps - self.warmup_steps - x) / (
                total_steps - self.warmup_steps
            )
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, linear_decay_func
            )
        elif opt == "exponential":
            lr_scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, **lr_scheduler_config.get("exponential_config", {})
            )
        elif opt == "step":
            lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, **lr_scheduler_config.get("step_config", {})
            )
        else:
            raise ValueError(f"Unrecognized lr scheduler option '{opt}'")

        self.lr_scheduler = lr_scheduler

    def _set_warmup_scheduler(self, model):
        """Set warmup learning rate scheduler for learning process."""

        if self.config["lr_scheduler_config"]["warmup_steps"]:
            warmup_steps = self.config["lr_scheduler_config"]["warmup_steps"]
            if warmup_steps < 0:
                raise ValueError(f"warmup_steps much greater or equal than 0.")
            warmup_unit = self.config["lr_scheduler_config"]["warmup_unit"]
            if warmup_unit == "epochs":
                self.warmup_steps = int(warmup_steps * self.n_batches_per_epoch)
            elif warmup_unit == "batches":
                self.warmup_steps = int(warmup_steps)
            else:
                raise ValueError(
                    f"warmup_unit must be 'batches' or 'epochs', but {warmup_unit} found."
                )
            linear_warmup_func = lambda x: x / self.warmup_steps
            warmup_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, linear_warmup_func
            )
            logging.info(f"Warmup {self.warmup_steps} batches.")
        elif self.config["lr_scheduler_config"]["warmup_percentage"]:
            warmup_percentage = self.config["lr_scheduler_config"]["warmup_percentage"]
            self.warmup_steps = int(
                warmup_percentage * self.config["n_epochs"] * self.n_batches_per_epoch
            )
            linear_warmup_func = lambda x: x / self.warmup_steps
            warmup_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, linear_warmup_func
            )
            logging.info(f"Warmup {self.warmup_steps} batches.")
        else:
            warmup_scheduler = None
            self.warmup_steps = 0

        self.warmup_scheduler = warmup_scheduler

    def _update_lr_scheduler(self, model, step):
        """Update the lr using lr_scheduler with each batch."""

        if self.warmup_scheduler and step < self.warmup_steps:
            self.warmup_scheduler.step()
        elif self.lr_scheduler is not None:
            self.lr_scheduler.step()
            min_lr = self.config["lr_scheduler_config"]["min_lr"]
            if min_lr and self.optimizer.param_groups[0]["lr"] < min_lr:
                self.optimizer.param_groups[0]["lr"] = min_lr

    def _set_batch_scheduler(self):
        """Set task scheduler for learning process"""
        scheduler_class = batch_schedulers.get(self.config["batch_scheduler"])
        if not scheduler_class:
            raise ValueError(f"Unrecognized batch scheduler option '{scheduler_class}'")

        self.batch_scheduler = scheduler_class()

    def _evaluate(self, model, dataloaders, split):
        return model.score([dl for dl in dataloaders if dl.dataset.split in split])

    def _logging(self, model, dataloaders, batch_size):
        """Checking if it's time to evaluting or checkpointing"""

        # Switch to eval mode for evaluation
        model.eval()

        self.log_manager.update(batch_size)

        # Log the loss and lr
        metric_dict = dict()
        metric_dict.update(self._aggregate_losses())

        # Evaluate the model and log the metric
        if self.log_manager.trigger_evaluation():

            # Log metrics
            metric_dict.update(
                self._evaluate(model, dataloaders, self.config["valid_split"])
            )

            if self.config["logging"]:
                self._log_metrics(metric_dict)

            self._reset_losses()

        # Checkpoint the model
        if self.log_manager.trigger_checkpointing():
            if self.config["checkpointing"]:
                self._checkpoint_model(model, metric_dict)

            self._reset_losses()

        # Switch back to train mode
        model.train()
        return metric_dict

    def _log_metrics(self, metric_dict):
        for metric_name, metric_value in metric_dict.items():
            self.log_writer.add_scalar(
                metric_name, metric_value, self.log_manager.point_total
            )

    def _checkpoint_model(self, model, metric_dict):
        self.checkpointer.checkpoint(
            self.log_manager.unit_total,
            model,
            self.optimizer,
            self.lr_scheduler,
            metric_dict,
        )

    def _aggregate_losses(self):
        """Calculate the task specific loss, average micro loss and learning rate."""

        metric_dict = dict()

        # Log task specific loss
        for identifier in self.running_losses.keys():
            if self.running_counts[identifier] > 0:
                metric_dict[identifier] = (
                    self.running_losses[identifier] / self.running_counts[identifier]
                )

        # Calculate average micro loss
        total_loss = sum(self.running_losses.values())
        total_count = sum(self.running_counts.values())
        if total_count > 0:
            metric_dict["model/all/train/loss"] = total_loss / total_count

        # Log the learning rate
        metric_dict["model/all/train/lr"] = self.optimizer.param_groups[0]["lr"]

        return metric_dict

    def _reset_losses(self):
        self.running_losses = defaultdict(float)
        self.running_counts = defaultdict(int)
