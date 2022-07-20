import logging
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from snorkel.classification.data import DictDataLoader  # noqa: F401
from snorkel.classification.multitask_classifier import (
    ClassifierConfig,
    MultitaskClassifier,
)
from snorkel.types import Config
from snorkel.utils.config_utils import merge_config
from snorkel.utils.lr_schedulers import LRSchedulerConfig
from snorkel.utils.optimizers import OptimizerConfig

from .loggers import (
    Checkpointer,
    CheckpointerConfig,
    LogManager,
    LogManagerConfig,
    LogWriter,
    LogWriterConfig,
    TensorBoardWriter,
)
from .schedulers import batch_schedulers

Metrics = Dict[str, float]


class TrainerConfig(Config):
    """Settings for the Trainer.

    Parameters
    ----------
    seed
        A random seed to set before training; if None, no seed is set
    n_epochs
        The number of epochs to train
    lr
        Base learning rate (will also be affected by lr_scheduler choice and settings)
    l2
        L2 regularization coefficient (weight decay)
    grad_clip
        The value that the gradient norm will be clipped to if it exceeds it
    train_split
        The name of the split to use as the training set
    valid_split
        The name of the split to use as the validation set
    test_split
        The name of the split to use as the test set
    progress_bar
        If True, print a tqdm progress bar during training
    model_config
        Settings for the MultitaskClassifier
    log_manager_config
        Settings for the LogManager
    checkpointing
        If True, use a Checkpointer to save the best model during training
    checkpointer_config
        Settings for the Checkpointer
    logging
        If True, log metrics (to file or Tensorboard) during training
    log_writer
        The type of LogWriter to use (one of ["json", "tensorboard"])
    log_writer_config
        Settings for the LogWriter
    optimizer
        Which optimizer to use (one of ["sgd", "adam", "adamax"])
    optimizer_config
        Settings for the optimizer
    lr_scheduler
        Which lr_scheduler to use (one of ["constant", "linear", "exponential", "step"])
    lr_scheduler_config
        Settings for the LRScheduler
    batch_scheduler
        Which batch scheduler to use (in what order batches will be drawn from multiple
        tasks)
    """

    seed: Optional[int] = None
    n_epochs: int = 1
    lr: float = 0.01
    l2: float = 0.0
    grad_clip: float = 1.0
    train_split: str = "train"
    valid_split: str = "valid"
    test_split: str = "test"
    progress_bar: bool = True
    model_config: ClassifierConfig = ClassifierConfig()  # type:ignore
    log_manager_config: LogManagerConfig = LogManagerConfig()  # type:ignore
    checkpointing: bool = False
    checkpointer_config: CheckpointerConfig = CheckpointerConfig()  # type:ignore
    logging: bool = False
    log_writer: str = "tensorboard"
    log_writer_config: LogWriterConfig = LogWriterConfig()  # type:ignore
    optimizer: str = "adam"
    optimizer_config: OptimizerConfig = OptimizerConfig()  # type:ignore
    lr_scheduler: str = "constant"
    lr_scheduler_config: LRSchedulerConfig = LRSchedulerConfig()  # type:ignore
    batch_scheduler: str = "shuffled"


class Trainer:
    """A class for training a MultitaskClassifier.

    Parameters
    ----------
    name
        An optional name for this trainer object
    kwargs
        Settings to be merged into the default Trainer config dict

    Attributes
    ----------
    name
        See above
    config
        The config dict with the settings for the Trainer
    checkpointer
        Saves the best model seen during training
    log_manager
        Identifies when its time to log or evaluate on the valid set
    log_writer
        Writes training statistics to file or TensorBoard
    optimizer
        Updates model weights based on the loss
    lr_scheduler
        Adjusts the learning rate over the course of training
    batch_scheduler
        Returns batches from the DataLoaders in a particular order for training
    """

    def __init__(self, name: Optional[str] = None, **kwargs: Any) -> None:
        self.config: TrainerConfig = merge_config(  # type:ignore
            TrainerConfig(), kwargs  # type:ignore
        )
        self.name = name if name is not None else type(self).__name__

    def fit(
        self, model: MultitaskClassifier, dataloaders: List["DictDataLoader"]
    ) -> None:
        """Train a MultitaskClassifier.

        Parameters
        ----------
        model
            The model to train
        dataloaders
            A list of DataLoaders. These will split into train, valid, and test splits
            based on the ``split`` attribute of the DataLoaders.
        """
        self._check_dataloaders(dataloaders)

        # Identify the dataloaders to train on
        train_dataloaders = [
            dl
            for dl in dataloaders
            if dl.dataset.split == self.config.train_split  # type: ignore
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
        self._set_lr_scheduler()
        self._set_batch_scheduler()

        # Set to training mode
        model.train()

        logging.info("Start training...")

        self.metrics: Dict[str, float] = dict()
        self._reset_losses()

        for epoch_num in range(self.config.n_epochs):
            batches = tqdm(
                enumerate(self.batch_scheduler.get_batches(train_dataloaders)),
                total=self.n_batches_per_epoch,
                disable=(not self.config.progress_bar),
                desc=f"Epoch {epoch_num}:",
            )
            for batch_num, (batch, dataloader) in batches:
                X_dict, Y_dict = batch

                total_batch_num = epoch_num * self.n_batches_per_epoch + batch_num
                batch_size = len(next(iter(Y_dict.values())))

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
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.grad_clip
                    )

                # Update the parameters
                self.optimizer.step()

                # Update lr using lr scheduler
                self._update_lr_scheduler(total_batch_num)

                # Update metrics
                self.metrics.update(self._logging(model, dataloaders, batch_size))

                batches.set_postfix(self.metrics)

        model = self.log_manager.cleanup(model)

    def _check_dataloaders(self, dataloaders: List["DictDataLoader"]) -> None:
        """Validate the dataloader splits."""
        train_split = self.config.train_split
        valid_split = self.config.valid_split
        test_split = self.config.test_split

        all_splits = [train_split, valid_split, test_split]
        if not all(d.dataset.split in all_splits for d in dataloaders):  # type: ignore
            raise ValueError(f"Dataloader splits must be one of {all_splits}")

        if not any(d.dataset.split == train_split for d in dataloaders):  # type: ignore
            raise ValueError(
                f"Cannot find any dataloaders with split matching train split: "
                f"{self.config.train_split}."
            )

    def _set_log_writer(self) -> None:
        self.log_writer: Optional[LogWriter] = None
        if self.config.logging:
            if self.config.log_writer == "json":
                self.log_writer = LogWriter(**self.config.log_writer_config._asdict())
            elif self.config.log_writer == "tensorboard":
                self.log_writer = TensorBoardWriter(
                    **self.config.log_writer_config._asdict()
                )
            else:
                raise ValueError(
                    f"Unrecognized writer option: {self.config.log_writer}"
                )

    def _set_checkpointer(self) -> None:
        self.checkpointer: Optional[Checkpointer]

        if self.config.checkpointing:
            checkpointer_config = self.config.checkpointer_config
            evaluation_freq = self.config.log_manager_config.evaluation_freq
            counter_unit = self.config.log_manager_config.counter_unit
            self.checkpointer = Checkpointer(
                counter_unit, evaluation_freq, **checkpointer_config._asdict()
            )
        else:
            self.checkpointer = None

    def _set_log_manager(self) -> None:
        self.log_manager = LogManager(
            self.n_batches_per_epoch,
            log_writer=self.log_writer,
            checkpointer=self.checkpointer,
            **self.config.log_manager_config._asdict(),
        )

    def _set_optimizer(self, model: nn.Module) -> None:
        optimizer_config = self.config.optimizer_config
        optimizer_name = self.config.optimizer

        parameters = filter(lambda p: p.requires_grad, model.parameters())

        optimizer: optim.Optimizer  # type: ignore

        if optimizer_name == "sgd":
            optimizer = optim.SGD(  # type: ignore
                parameters,
                lr=self.config.lr,
                weight_decay=self.config.l2,
                **optimizer_config.sgd_config._asdict(),
            )
        elif optimizer_name == "adam":
            optimizer = optim.Adam(
                parameters,
                lr=self.config.lr,
                weight_decay=self.config.l2,
                **optimizer_config.adam_config._asdict(),
            )
        elif optimizer_name == "adamax":
            optimizer = optim.Adamax(  # type: ignore
                parameters,
                lr=self.config.lr,
                weight_decay=self.config.l2,
                **optimizer_config.adamax_config._asdict(),
            )
        else:
            raise ValueError(f"Unrecognized optimizer option '{optimizer_name}'")

        logging.info(f"Using optimizer {optimizer}")

        self.optimizer = optimizer

    def _set_lr_scheduler(self) -> None:
        # Set warmup scheduler
        self._set_warmup_scheduler()

        # Set lr scheduler
        lr_scheduler_name = self.config.lr_scheduler
        lr_scheduler_config = self.config.lr_scheduler_config
        lr_scheduler: Optional[optim.lr_scheduler._LRScheduler]

        if lr_scheduler_name == "constant":
            lr_scheduler = None
        elif lr_scheduler_name == "linear":
            total_steps = self.n_batches_per_epoch * self.config.n_epochs
            linear_decay_func = lambda x: (total_steps - self.warmup_steps - x) / (
                total_steps - self.warmup_steps
            )
            lr_scheduler = optim.lr_scheduler.LambdaLR(  # type: ignore
                self.optimizer, linear_decay_func
            )
        elif lr_scheduler_name == "exponential":
            lr_scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, **lr_scheduler_config.exponential_config._asdict()
            )
        elif lr_scheduler_name == "step":
            lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, **lr_scheduler_config.step_config._asdict()
            )
        else:
            raise ValueError(f"Unrecognized lr scheduler option '{lr_scheduler_name}'")

        self.lr_scheduler = lr_scheduler

    def _set_warmup_scheduler(self) -> None:
        warmup_scheduler: Optional[optim.lr_scheduler.LambdaLR]

        if self.config.lr_scheduler_config.warmup_steps:
            warmup_steps = self.config.lr_scheduler_config.warmup_steps
            if warmup_steps < 0:
                raise ValueError("warmup_steps much greater or equal than 0.")
            warmup_unit = self.config.lr_scheduler_config.warmup_unit
            if warmup_unit == "epochs":
                self.warmup_steps = int(warmup_steps * self.n_batches_per_epoch)
            elif warmup_unit == "batches":
                self.warmup_steps = int(warmup_steps)
            else:
                raise ValueError(
                    f"warmup_unit must be 'batches' or 'epochs', but {warmup_unit} found."
                )
            linear_warmup_func = lambda x: x / self.warmup_steps
            warmup_scheduler = optim.lr_scheduler.LambdaLR(  # type: ignore
                self.optimizer, linear_warmup_func
            )
            logging.info(f"Warmup {self.warmup_steps} batches.")
        elif self.config.lr_scheduler_config.warmup_percentage:
            warmup_percentage = self.config.lr_scheduler_config.warmup_percentage
            self.warmup_steps = int(
                warmup_percentage * self.config.n_epochs * self.n_batches_per_epoch
            )
            linear_warmup_func = lambda x: x / self.warmup_steps
            warmup_scheduler = optim.lr_scheduler.LambdaLR(  # type: ignore
                self.optimizer, linear_warmup_func
            )
            logging.info(f"Warmup {self.warmup_steps} batches.")
        else:
            warmup_scheduler = None
            self.warmup_steps = 0

        self.warmup_scheduler = warmup_scheduler

    def _update_lr_scheduler(self, step: int) -> None:
        if self.warmup_scheduler and step < self.warmup_steps:
            self.warmup_scheduler.step()  # type: ignore
        elif self.lr_scheduler is not None:
            self.lr_scheduler.step()  # type: ignore
            min_lr = self.config.lr_scheduler_config.min_lr
            if min_lr and self.optimizer.param_groups[0]["lr"] < min_lr:
                self.optimizer.param_groups[0]["lr"] = min_lr

    def _set_batch_scheduler(self) -> None:
        scheduler_class = batch_schedulers.get(self.config.batch_scheduler)
        if not scheduler_class:
            raise ValueError(f"Unrecognized batch scheduler option '{scheduler_class}'")

        self.batch_scheduler = scheduler_class()  # type: ignore

    def _evaluate(
        self,
        model: MultitaskClassifier,
        dataloaders: List["DictDataLoader"],
        split: str,
    ) -> Metrics:
        """Evalute the current quality of the model on data for the requested split."""
        loaders = [d for d in dataloaders if d.dataset.split in split]  # type: ignore
        return model.score(loaders)

    def _logging(
        self,
        model: MultitaskClassifier,
        dataloaders: List["DictDataLoader"],
        batch_size: int,
    ) -> Metrics:
        """Log and checkpoint if it is time to do so."""

        # Switch to eval mode for evaluation
        model.eval()

        self.log_manager.update(batch_size)

        # Log the loss and lr
        metric_dict: Metrics = dict()
        metric_dict.update(self._aggregate_losses())

        # Evaluate the model and log the metric
        if self.log_manager.trigger_evaluation():

            # Log metrics
            metric_dict.update(
                self._evaluate(model, dataloaders, self.config.valid_split)
            )
            self._log_metrics(metric_dict)
            self._reset_losses()

        # Checkpoint the model
        if self.log_manager.trigger_checkpointing():
            self._checkpoint_model(model, metric_dict)
            self._reset_losses()

        # Switch back to train mode
        model.train()
        return metric_dict

    def _log_metrics(self, metric_dict: Metrics) -> None:
        if self.log_writer is not None:
            for metric_name, metric_value in metric_dict.items():
                self.log_writer.add_scalar(
                    metric_name, metric_value, self.log_manager.point_total
                )

    def _checkpoint_model(
        self, model: MultitaskClassifier, metric_dict: Metrics
    ) -> None:
        """Save the current model."""
        if self.checkpointer is not None:
            self.checkpointer.checkpoint(
                self.log_manager.unit_total, model, metric_dict
            )

    def _aggregate_losses(self) -> Metrics:
        """Calculate the task specific loss, average micro loss and learning rate."""

        metric_dict = dict()

        # Log task specific loss
        self.running_losses: DefaultDict[str, float]
        self.running_counts: DefaultDict[str, float]
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

    def _reset_losses(self) -> None:
        """Reset the loss counters."""
        self.running_losses = defaultdict(float)
        self.running_counts = defaultdict(int)

    def save(self, trainer_path: str) -> None:
        """Save the trainer config to the specified file path in json format.

        Parameters
        ----------
        trainer_path
            The path where trainer config and optimizer state should be saved.
        """

        head, tail = os.path.split(trainer_path)

        if not os.path.exists(head):
            os.makedirs(os.path.dirname(head))
        try:
            torch.save(
                {
                    "trainer_config": self.config._asdict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                trainer_path,
            )
        except BaseException:  # pragma: no cover
            logging.warning("Saving failed... continuing anyway.")

        logging.info(f"[{self.name}] Trainer config saved in {trainer_path}")

    def load(self, trainer_path: str, model: Optional[MultitaskClassifier]) -> None:
        """Load trainer config and optimizer state from the specified json file path to the trainer object. The optimizer state is stored, too. However, it only makes sense if loaded with the correct model again.

        Parameters
        ----------
        trainer_path
            The path to the saved trainer config to be loaded
        model
            MultitaskClassifier for which the optimizer has been set. Parameters of optimizer must fit to model parameters. This model
            shall be the model which was fit by the stored Trainer.

        Example
        -------
        Saving model and corresponding trainer:
        >>> model.save('./my_saved_model_file') # doctest: +SKIP
        >>> trainer.save('./my_saved_trainer_file') # doctest: +SKIP
        Now we can resume training and load the saved model and trainer into new model and trainer objects:
        >>> new_model.load('./my_saved_model_file') # doctest: +SKIP
        >>> new_trainer.load('./my_saved_trainer_file', model=new_model) # doctest: +SKIP
        >>> new_trainer.fit(...) # doctest: +SKIP
        """

        try:
            saved_state = torch.load(trainer_path)
        except BaseException:
            if not os.path.exists(trainer_path):
                logging.error("Loading failed... Trainer config does not exist.")
            else:
                logging.error(
                    f"Loading failed... Cannot load trainer config from {trainer_path}"
                )
            raise

        self.config = TrainerConfig(**saved_state["trainer_config"])
        logging.info(f"[{self.name}] Trainer config loaded from {trainer_path}")

        if model is not None:
            try:
                self._set_optimizer(model)
                self.optimizer.load_state_dict(saved_state["optimizer_state_dict"])
                logging.info(f"[{self.name}] Optimizer loaded from {trainer_path}")
            except BaseException:
                logging.error(
                    "Loading the optimizer for your model failed. Optimizer state NOT loaded."
                )
