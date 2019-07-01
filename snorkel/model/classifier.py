import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset, TensorDataset

from snorkel.analysis.error_analysis import confusion_matrix
from snorkel.analysis.metrics import metric_score

from .logging import Checkpointer, Logger, LogWriter, TensorBoardWriter
from .utils import place_on_gpu, recursive_merge_dicts, set_seed

# Import tqdm_notebook if in Jupyter notebook
try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    # Only use tqdm notebook if not in travis testing
    if "CI" not in os.environ:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm


class Classifier(nn.Module):
    """Simple abstract base class for a probabilistic classifier.

    The main contribution of children classes will be an implementation of the
    predict_proba() method. The relationships between the predict/score
    functions are as follows:

    score
        |
    predict
        |
    *predict_proba

    The method predict_proba() method calculates the probabilistic labels,
    the predict() method handles tie-breaking, and the score() method
    calculates metrics based on predictions.

    Args:
        k: (int) The cardinality of the classifier
        config: (dict) A config dictionary
    """

    # A class variable indicating whether the class implements its own custom L2
    # regularization (True) or not (False); in the latter case, generic L2 in
    # the optimizer is used
    implements_l2 = False

    def __init__(self, k, config):
        super().__init__()
        self.config = config
        self.multitask = False
        self.k = k

        # Set random seed
        if self.config["seed"] is None:
            self.config["seed"] = np.random.randint(1e6)
        self.seed = self.config["seed"]
        set_seed(self.seed)

        # Confirm that cuda is available if config is using CUDA
        if self.config["device"] != "cpu" and not torch.cuda.is_available():
            raise ValueError("device=cuda but CUDA not available.")

        # By default, put model in eval mode; switch to train mode in training
        self.eval()

    def predict_proba(self, X, **kwargs):
        """Predicts probabilistic labels for an input X on all tasks
        Args:
            X: An appropriate input for the child class of Classifier
        Returns:
            An [n, k] np.ndarray of probabilities
        """
        raise NotImplementedError

    def predict(self, X, break_ties="random", return_probs=False, **kwargs):
        """Predicts (int) labels for an input X on all tasks

        Args:
            X: The input for the predict_proba method
            break_ties: A tie-breaking policy (see Classifier._break_ties())
            return_probs: Return the predicted probabilities as well

        Returns:
            Y_p: An n-dim np.ndarray of predictions in {1,...k}
            [Optionally: Y_s: An [n, k] np.ndarray of predicted probabilities]
        """
        Y_s = self._to_numpy(self.predict_proba(X, **kwargs))
        Y_p = self._break_ties(Y_s, break_ties).astype(np.int)
        if return_probs:
            return Y_p, Y_s
        else:
            return Y_p

    def score(
        self,
        data,
        metric="accuracy",
        break_ties="random",
        verbose=True,
        print_confusion_matrix=True,
        **kwargs,
    ):
        """Scores the predictive performance of the Classifier on all tasks

        Args:
            data: a Pytorch DataLoader, Dataset, or tuple with Tensors (X,Y):
                X: The input for the predict method
                Y: An [n] or [n, 1] torch.Tensor or np.ndarray of target labels
                    in {1,...,k}
            metric: A metric (string) with which to score performance or a
                list of such metrics
            break_ties: A tie-breaking policy (see Classifier._break_ties())
            verbose: The verbosity for just this score method; it will not
                update the class config.
            print_confusion_matrix: Print confusion matrix (overwritten to False if
                verbose=False)

        Returns:
            scores: A (float) score or a list of such scores if kwarg metric is a list
        """
        Y_p, Y, Y_s = self._get_predictions(
            data, break_ties=break_ties, return_probs=True, **kwargs
        )

        # Evaluate on the specified metrics
        return_list = isinstance(metric, list)
        metric_list = metric if isinstance(metric, list) else [metric]
        scores = []
        for metric in metric_list:
            score = metric_score(
                gold=Y, pred=Y_p, prob=Y_s, metric=metric, ignore_in_gold=[0]
            )
            scores.append(score)
            if verbose:
                print(f"{metric.capitalize()}: {score:.3f}")

        # Optionally print confusion matrix
        if print_confusion_matrix and verbose:
            confusion_matrix(Y, Y_p, pretty_print=True)

        # If a single metric was given as a string (not list), return a float
        if len(scores) == 1 and not return_list:
            return scores[0]
        else:
            return scores

    def train_model(self, *args, **kwargs):
        """Trains a classifier

        Take care to initialize weights outside the training loop and zero out
        gradients at the beginning of each iteration inside the loop.

        NOTE: self.train() is a method in nn.Module class, so we name this
        method `train_model` so as not to conflict.
        """
        raise NotImplementedError

    def _train_model(
        self, train_data, loss_fn, valid_data=None, log_writer=None, restore_state={}
    ):
        """The internal training routine called by train_model() after setup

        Args:
            train_data: a tuple of Tensors (X,Y), a Dataset, or a DataLoader of
                X (data) and Y (labels) for the train split
            loss_fn: the loss function to minimize (maps *data -> loss)
            valid_data: a tuple of Tensors (X,Y), a Dataset, or a DataLoader of
                X (data) and Y (labels) for the dev split
            restore_state: a dictionary containing model weights (optimizer, main network) and training information

        If valid_data is not provided, then no checkpointing or
        evaluation on the dev set will occur.
        """
        # Set model to train mode
        self.train()
        train_config = self.config["train_config"]

        # Convert data to DataLoaders
        train_loader = self._create_data_loader(train_data)
        valid_loader = self._create_data_loader(valid_data)
        epoch_size = len(train_loader.dataset)

        # Move model to GPU
        if self.config["verbose"] and self.config["device"] != "cpu":
            print("Using GPU...")
        self.to(self.config["device"])

        # Set training components
        self._set_writer(train_config)
        self._set_logger(train_config, epoch_size)
        self._set_checkpointer(train_config)
        self._set_optimizer(train_config)
        self._set_scheduler(train_config)

        # Restore model if necessary
        if restore_state:
            start_iteration = self._restore_training_state(restore_state)
        else:
            start_iteration = 0

        # Train the model
        metrics_hist = {}  # The most recently seen value for all metrics
        for epoch in range(start_iteration, train_config["n_epochs"]):
            progress_bar = (
                train_config["progress_bar"]
                and self.config["verbose"]
                and self.logger.log_unit == "epochs"
            )

            t = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                disable=(not progress_bar),
            )

            self.running_loss = 0.0
            self.running_examples = 0
            for batch_num, data in t:
                # NOTE: actual batch_size may not equal config's target batch_size
                batch_size = len(data[0])

                # Moving data to device
                if self.config["device"] != "cpu":
                    data = place_on_gpu(data)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass to calculate the average loss per example
                loss = loss_fn(*data)
                if torch.isnan(loss):
                    msg = "Loss is NaN. Consider reducing learning rate."
                    raise Exception(msg)

                # Backward pass to calculate gradients
                # Loss is an average loss per example
                loss.backward()

                # Perform optimizer step
                self.optimizer.step()

                # Calculate metrics, log, and checkpoint as necessary
                metrics_dict = self._execute_logging(
                    train_loader, valid_loader, loss, batch_size
                )
                metrics_hist.update(metrics_dict)

                # tqdm output
                t.set_postfix(loss=metrics_dict["train/loss"])

            # Apply learning rate scheduler
            self._update_scheduler(epoch, metrics_hist)

        self.eval()

        # Restore best model if applicable
        if self.checkpointer and self.checkpointer.checkpoint_best:
            self.checkpointer.load_best_model(model=self)

        # Write log if applicable
        if self.writer:
            if self.writer.include_config:
                self.writer.add_config(self.config)
            self.writer.close()

        # Print confusion matrix if applicable
        if self.config["verbose"]:
            print("Finished Training")
            if valid_loader is not None:
                self.score(
                    valid_loader,
                    metric=train_config["validation_metric"],
                    verbose=True,
                    print_confusion_matrix=True,
                )

    def _get_loss_fn(self):
        """Returns a loss function"""
        msg = (
            "Abstract class: _get_loss_fn() must be implemented by a child "
            "class of Classifier."
        )
        raise NotImplementedError(msg)

    def save(self, destination, **kwargs):
        """Serialize and save a model.

        Example:
            end_model = EndModel(...)
            end_model.train_model(...)
            end_model.save("my_end_model.pkl")
        """
        with open(destination, "wb") as f:
            torch.save(self, f, **kwargs)

    @staticmethod
    def load(source, **kwargs):
        """Deserialize and load a model.

        Example:
            end_model = EndModel.load("my_end_model.pkl")
            end_model.score(...)
        """
        with open(source, "rb") as f:
            return torch.load(f, **kwargs)

    def update_config(self, update_dict):
        """Updates self.config with the values in a given update dictionary"""
        self.config = recursive_merge_dicts(self.config, update_dict)

    def reset(self):
        """Initializes all modules in a network"""
        # The apply(f) method recursively calls f on itself and all children
        self.apply(self._reset_module)

    @staticmethod
    def _reset_module(m):
        """An initialization method to be applied recursively to all modules"""
        raise NotImplementedError

    def resume_training(self, train_data, model_path, valid_data=None):
        """This model resume training of a classifier by reloading the appropriate state_dicts for each model

        Args:
           train_data: a tuple of Tensors (X,Y), a Dataset, or a DataLoader of
                X (data) and Y (labels) for the train split
            model_path: the path to the saved checpoint for resuming training
            valid_data: a tuple of Tensors (X,Y), a Dataset, or a DataLoader of
                X (data) and Y (labels) for the dev split
        """
        restore_state = self.checkpointer.restore(model_path)
        loss_fn = self._get_loss_fn()
        self.train()
        self._train_model(
            train_data=train_data,
            loss_fn=loss_fn,
            valid_data=valid_data,
            restore_state=restore_state,
        )

    def _restore_training_state(self, restore_state):
        """Restores the model and optimizer states

        This helper function restores the model's state to a given iteration so
        that a user can resume training at any epoch.

        Args:
            restore_state: a state_dict dictionary
        """
        self.load_state_dict(restore_state["model"])
        self.optimizer.load_state_dict(restore_state["optimizer"])
        self.lr_scheduler.load_state_dict(restore_state["lr_scheduler"])
        start_iteration = restore_state["iteration"] + 1
        if self.config["verbose"]:
            print(f"Restored checkpoint to iteration {start_iteration}.")

        if restore_state["best_model_found"]:
            # Update checkpointer with appropriate information about best model
            # Note that the best model found so far may not be the model in the
            # checkpoint that is currently being loaded.
            self.checkpointer.best_model_found = True
            self.checkpointer.best_iteration = restore_state["best_iteration"]
            self.checkpointer.best_score = restore_state["best_score"]
            if self.config["verbose"]:
                print(
                    f"Updated checkpointer: "
                    f"best_score={self.checkpointer.best_score:.3f}, "
                    f"best_iteration={self.checkpointer.best_iteration}"
                )
        return start_iteration

    def _create_dataset(self, *data):
        """Converts input data to the appropriate Dataset"""
        # Make sure data is a tuple of dense tensors
        data = [self._to_torch(x, dtype=torch.FloatTensor) for x in data]
        return TensorDataset(*data)

    def _create_data_loader(self, data, **kwargs):
        """Converts input data into a DataLoader"""
        if data is None:
            return None

        # Set DataLoader config
        # NOTE: Not applicable if data is already a DataLoader
        config = {
            **self.config["train_config"]["data_loader_config"],
            **kwargs,
            "pin_memory": self.config["device"] != "cpu",
        }
        # Return data as DataLoader
        if isinstance(data, DataLoader):
            return data
        elif isinstance(data, Dataset):
            return DataLoader(data, **config)
        elif isinstance(data, (tuple, list)):
            return DataLoader(self._create_dataset(*data), **config)
        else:
            raise ValueError("Input data type not recognized.")

    def _set_writer(self, train_config):
        if train_config["writer"] is None:
            self.writer = None
        elif train_config["writer"] == "json":
            self.writer = LogWriter(**(train_config["writer_config"]))
        elif train_config["writer"] == "tensorboard":
            self.writer = TensorBoardWriter(**(train_config["writer_config"]))
        else:
            raise Exception(f"Unrecognized writer: {train_config['writer']}")

    def _set_logger(self, train_config, epoch_size):
        self.logger = Logger(
            train_config["logger_config"],
            self.writer,
            epoch_size,
            verbose=self.config["verbose"],
        )

    def _set_checkpointer(self, train_config):
        if train_config["checkpoint"]:
            # Default to valid split for checkpoint metric
            checkpoint_config = train_config["checkpoint_config"]
            checkpoint_metric = checkpoint_config["checkpoint_metric"]
            if checkpoint_metric.count("/") == 0:
                checkpoint_config["checkpoint_metric"] = f"valid/{checkpoint_metric}"
            self.checkpointer = Checkpointer(
                checkpoint_config, verbose=self.config["verbose"]
            )
        else:
            self.checkpointer = None

    def _set_optimizer(self, train_config):
        optimizer_config = train_config["optimizer_config"]
        opt = optimizer_config["optimizer"]

        # We set L2 here if the class does not implement its own L2 reg
        l2 = 0 if self.implements_l2 else train_config.get("l2", 0)

        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if opt == "sgd":
            optimizer = optim.SGD(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["sgd_config"],
                weight_decay=l2,
            )
        elif opt == "rmsprop":
            optimizer = optim.RMSprop(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["rmsprop_config"],
                weight_decay=l2,
            )
        elif opt == "adam":
            optimizer = optim.Adam(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["adam_config"],
                weight_decay=l2,
            )
        elif opt == "sparseadam":
            optimizer = optim.SparseAdam(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["adam_config"],
            )
            if l2:
                raise Exception(
                    "SparseAdam optimizer does not support weight_decay (l2 penalty)."
                )
        else:
            raise ValueError(f"Did not recognize optimizer option '{opt}'")
        self.optimizer = optimizer

    def _set_scheduler(self, train_config):
        lr_scheduler = train_config["lr_scheduler"]
        if lr_scheduler is None:
            lr_scheduler = None
        else:
            lr_scheduler_config = train_config["lr_scheduler_config"]
            if lr_scheduler == "exponential":
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, **lr_scheduler_config["exponential_config"]
                )
            elif lr_scheduler == "reduce_on_plateau":
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, **lr_scheduler_config["plateau_config"]
                )
            else:
                raise ValueError(
                    f"Did not recognize lr_scheduler option '{lr_scheduler}'"
                )
        self.lr_scheduler = lr_scheduler

    def _update_scheduler(self, epoch, metrics_dict):
        train_config = self.config["train_config"]
        if self.lr_scheduler is not None:
            lr_scheduler_config = train_config["lr_scheduler_config"]
            if epoch + 1 >= lr_scheduler_config["lr_freeze"]:
                if train_config["lr_scheduler"] == "reduce_on_plateau":
                    checkpoint_config = train_config["checkpoint_config"]
                    metric_name = checkpoint_config["checkpoint_metric"]
                    score = metrics_dict.get(metric_name, None)
                    if score is not None:
                        self.lr_scheduler.step(score)
                else:
                    self.lr_scheduler.step()

    def _execute_logging(self, train_loader, valid_loader, loss, batch_size):
        self.eval()
        self.running_loss += loss.item() * batch_size
        self.running_examples += batch_size

        # Initialize metrics dict
        metrics_dict = {}
        # Always add average loss
        metrics_dict["train/loss"] = self.running_loss / self.running_examples

        if self.logger.check(batch_size):
            logger_metrics = self.logger.calculate_metrics(
                self, train_loader, valid_loader, metrics_dict
            )
            metrics_dict.update(logger_metrics)
            self.logger.log(metrics_dict)

            # Reset running loss and examples counts
            self.running_loss = 0.0
            self.running_examples = 0

        # Checkpoint if applicable
        self._checkpoint(metrics_dict)

        self.train()
        return metrics_dict

    def _checkpoint(self, metrics_dict):
        if self.checkpointer is None:
            return
        iteration = self.logger.unit_total
        self.checkpointer.checkpoint(
            metrics_dict, iteration, self, self.optimizer, self.lr_scheduler
        )

    def _get_predictions(self, data, break_ties="random", return_probs=False, **kwargs):
        """Computes predictions in batch, given a labeled dataset

        Args:
            data: a Pytorch DataLoader, Dataset, or tuple with Tensors (X,Y):
                X: The input for the predict method
                Y: An [n] or [n, 1] torch.Tensor or np.ndarray of target labels
                    in {1,...,k}
            break_ties: How to break ties when making predictions
            return_probs: Return the predicted probabilities as well

        Returns:
            Y_p: A Tensor of predictions
            Y: A Tensor of labels
            [Optionally: Y_s: An [n, k] np.ndarray of predicted probabilities]
        """
        data_loader = self._create_data_loader(data)
        Y_p = []
        Y = []
        Y_s = []

        # Do batch evaluation by default, getting the predictions and labels
        for batch_num, data in enumerate(data_loader):
            Xb, Yb = data
            Y.append(self._to_numpy(Yb))

            # Optionally move to device
            if self.config["device"] != "cpu":
                Xb = place_on_gpu(Xb)

            # Append predictions and labels from DataLoader
            Y_pb, Y_sb = self.predict(
                Xb, break_ties=break_ties, return_probs=True, **kwargs
            )
            Y_p.append(self._to_numpy(Y_pb))
            Y_s.append(self._to_numpy(Y_sb))
        Y_p, Y, Y_s = map(self._stack_batches, [Y_p, Y, Y_s])
        if return_probs:
            return Y_p, Y, Y_s
        else:
            return Y_p, Y

    def _break_ties(self, Y_s, break_ties="random"):
        """Break ties in each row of a tensor according to the specified policy

        Args:
            Y_s: An [n, k] np.ndarray of probabilities
            break_ties: A tie-breaking policy:
                "abstain": return an abstain vote (0)
                "random": randomly choose among the tied options
                    NOTE: if break_ties="random", repeated runs may have
                    slightly different results due to difference in broken ties
                [int]: ties will be broken by using this label
        """
        n, k = Y_s.shape
        Y_h = np.zeros(n)
        diffs = np.abs(Y_s - Y_s.max(axis=1).reshape(-1, 1))

        TOL = 1e-5
        for i in range(n):
            max_idxs = np.where(diffs[i, :] < TOL)[0]
            if len(max_idxs) == 1:
                Y_h[i] = max_idxs[0] + 1
            # Deal with "tie votes" according to the specified policy
            elif break_ties == "random":
                Y_h[i] = np.random.choice(max_idxs) + 1
            elif break_ties == "abstain":
                Y_h[i] = 0
            elif isinstance(break_ties, int):
                Y_h[i] = break_ties
            else:
                ValueError(f"break_ties={break_ties} policy not recognized.")
        return Y_h

    @staticmethod
    def _to_numpy(Z):
        """Converts a None, list, np.ndarray, or torch.Tensor to np.ndarray;
        also handles converting sparse input to dense."""
        if Z is None:
            return Z
        elif issparse(Z):
            return Z.toarray()
        elif isinstance(Z, np.ndarray):
            return Z
        elif isinstance(Z, list):
            return np.array(Z)
        elif isinstance(Z, torch.Tensor):
            return Z.cpu().numpy()
        else:
            msg = (
                f"Expected None, list, numpy.ndarray or torch.Tensor, "
                f"got {type(Z)} instead."
            )
            raise Exception(msg)

    @staticmethod
    def _to_torch(Z, dtype=None):
        """Converts a None, list, np.ndarray, or torch.Tensor to torch.Tensor;
        also handles converting sparse input to dense."""
        if Z is None:
            return None
        elif issparse(Z):
            Z = torch.from_numpy(Z.toarray())
        elif isinstance(Z, torch.Tensor):
            pass
        elif isinstance(Z, list):
            Z = torch.from_numpy(np.array(Z))
        elif isinstance(Z, np.ndarray):
            Z = torch.from_numpy(Z)
        else:
            msg = (
                f"Expected list, numpy.ndarray or torch.Tensor, "
                f"got {type(Z)} instead."
            )
            raise Exception(msg)

        return Z.type(dtype) if dtype else Z

    def _check(self, var, val=None, typ=None, shape=None):
        if val is not None and not var != val:
            msg = f"Expected value {val} but got value {var}."
            raise ValueError(msg)
        if typ is not None and not isinstance(var, typ):
            msg = f"Expected type {typ} but got type {type(var)}."
            raise ValueError(msg)
        if shape is not None and not var.shape != shape:
            msg = f"Expected shape {shape} but got shape {var.shape}."
            raise ValueError(msg)

    def _check_or_set_attr(self, name, val, set_val=False):
        if set_val:
            setattr(self, name, val)
        else:
            true_val = getattr(self, name)
            if val != true_val:
                raise Exception(f"{name} = {val}, but should be {true_val}.")

    @staticmethod
    def _stack_batches(X):
        """Stack a list of np.ndarrays along the first axis, returning an
        np.ndarray; note this is mainly for smooth hanlding of the multi-task
        setting."""
        X = [Classifier._to_numpy(Xb) for Xb in X]
        if len(X[0].shape) == 1:
            return np.hstack(X)
        elif len(X[0].shape) == 2:
            return np.vstack(X)
        else:
            raise ValueError(f"Can't stack {len(X[0].shape)}-dim batches.")
