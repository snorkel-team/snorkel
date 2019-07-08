from collections import Counter
from functools import partial
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from .graph_utils import get_clique_tree
from .lm_defaults import lm_default_config
from .logging import Checkpointer, Logger, LogWriter, TensorBoardWriter
from .utils import MetalDataset, place_on_gpu, recursive_merge_dicts, set_seed


def to_numpy(Z):
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


def to_torch(Z, dtype=None):
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
            f"Expected list, numpy.ndarray or torch.Tensor, " f"got {type(Z)} instead."
        )
        raise Exception(msg)

    return Z.type(dtype) if dtype else Z


def stack_batches(X):
    """Stack a list of np.ndarrays along the first axis, returning an
    np.ndarray; note this is mainly for smooth hanlding of the multi-task
    setting."""
    X = [to_numpy(Xb) for Xb in X]
    if len(X[0].shape) == 1:
        return np.hstack(X)
    elif len(X[0].shape) == 2:
        return np.vstack(X)
    else:
        raise ValueError(f"Can't stack {len(X[0].shape)}-dim batches.")


class LabelModel(nn.Module):
    """A conditionally independent LabelModel to learn labeling function accuracies and assign probabilistic labels

    Args:
        k: (int) the cardinality of the classifier
    """

    def __init__(self, k=2, **kwargs):
        super().__init__()
        self.config = recursive_merge_dicts(lm_default_config, kwargs)
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

    def _check_L(self, L):
        """Run some basic checks on L."""
        if issparse(L):
            L = L.todense()

        # Check for correct values, e.g. warning if in {-1,0,1}
        if np.any(L < 0):
            raise ValueError("L must have values in {0,1,...,k}.")

        return L

    def _create_L_ind(self, L):
        """Convert a label matrix with labels in 0...k to a one-hot format

        Args:
            L: An [n,m] scipy.sparse label matrix with values in {0,1,...,k}

        Returns:
            L_ind: An [n,m*k] dense np.ndarray with values in {0,1}

        Note that no column is required for 0 (abstain) labels.
        """

        L_ind = np.zeros((self.n, self.m * self.k))
        for y in range(1, self.k + 1):
            # A[x::y] slices A starting at x at intervals of y
            # e.g., np.arange(9)[0::3] == np.array([0,3,6])
            L_ind[:, (y - 1) :: self.k] = np.where(L == y, 1, 0)
        return L_ind

    def _get_augmented_label_matrix(self, L, higher_order=False):
        """Returns an augmented version of L where each column is an indicator
        for whether a certain source or clique of sources voted in a certain
        pattern.

        Args:
            L: An [n,m] scipy.sparse label matrix with values in {0,1,...,k}
        """
        # Create a helper data structure which maps cliques (as tuples of member
        # sources) --> {start_index, end_index, maximal_cliques}, where
        # the last value is a set of indices in this data structure
        self.c_data = {}
        for i in range(self.m):
            self.c_data[i] = {
                "start_index": i * self.k,
                "end_index": (i + 1) * self.k,
                "max_cliques": set(
                    [
                        j
                        for j in self.c_tree.nodes()
                        if i in self.c_tree.node[j]["members"]
                    ]
                ),
            }

        L_ind = self._create_L_ind(L)

        # Get the higher-order clique statistics based on the clique tree
        # First, iterate over the maximal cliques (nodes of c_tree) and
        # separator sets (edges of c_tree)
        if higher_order:
            L_aug = np.copy(L_ind)
            for item in chain(self.c_tree.nodes(), self.c_tree.edges()):
                if isinstance(item, int):
                    C = self.c_tree.node[item]
                elif isinstance(item, tuple):
                    C = self.c_tree[item[0]][item[1]]
                else:
                    raise ValueError(item)
                members = list(C["members"])

                # With unary maximal clique, just store its existing index
                C["start_index"] = members[0] * self.k
                C["end_index"] = (members[0] + 1) * self.k
            return L_aug
        else:
            return L_ind

    def _build_mask(self):
        """Build mask applied to O^{-1}, O for the matrix approx constraint"""
        self.mask = torch.ones(self.d, self.d).byte()
        for ci in self.c_data.values():
            si, ei = ci["start_index"], ci["end_index"]
            for cj in self.c_data.values():
                sj, ej = cj["start_index"], cj["end_index"]

                # Check if ci and cj are part of the same maximal clique
                # If so, mask out their corresponding blocks in O^{-1}
                if len(ci["max_cliques"].intersection(cj["max_cliques"])) > 0:
                    self.mask[si:ei, sj:ej] = 0
                    self.mask[sj:ej, si:ei] = 0

    def _generate_O(self, L, higher_order=False):
        """Form the overlaps matrix, which is just all the different observed
        combinations of values of pairs of sources

        Note that we only include the k non-abstain values of each source,
        otherwise the model not minimal --> leads to singular matrix
        """
        L_aug = self._get_augmented_label_matrix(L, higher_order=higher_order)
        self.d = L_aug.shape[1]
        self.O = torch.from_numpy(L_aug.T @ L_aug / self.n).float()

    def _init_params(self):
        """Initialize the learned params

        - \mu is the primary learned parameter, where each row corresponds to
        the probability of a clique C emitting a specific combination of labels,
        conditioned on different values of Y (for each column); that is:

            self.mu[i*self.k + j, y] = P(\lambda_i = j | Y = y)

        and similarly for higher-order cliques.
        """
        train_config = self.config["train_config"]

        # Initialize mu so as to break basic reflective symmetry
        # Note that we are given either a single or per-LF initial precision
        # value, prec_i = P(Y=y|\lf=y), and use:
        #   mu_init = P(\lf=y|Y=y) = P(\lf=y) * prec_i / P(Y=y)

        # Handle single values
        if isinstance(train_config["prec_init"], (int, float)):
            self._prec_init = train_config["prec_init"] * torch.ones(self.m)
        if self._prec_init.shape[0] != self.m:
            raise ValueError(f"prec_init must have shape {self.m}.")

        # Get the per-value labeling propensities
        # Note that self.O must have been computed already!
        lps = torch.diag(self.O).numpy()

        # TODO: Update for higher-order cliques!
        self.mu_init = torch.zeros(self.d, self.k)
        for i in range(self.m):
            for y in range(self.k):
                idx = i * self.k + y
                mu_init = torch.clamp(lps[idx] * self._prec_init[i] / self.p[y], 0, 1)
                self.mu_init[idx, y] += mu_init

        # Initialize randomly based on self.mu_init
        self.mu = nn.Parameter(self.mu_init.clone() * np.random.random()).float()

        # Build the mask over O^{-1}
        self._build_mask()

    def get_conditional_probs(self, source=None):
        """Returns the full conditional probabilities table as a numpy array,
        where row i*(k+1) + ly is the conditional probabilities of source i
        emmiting label ly (including abstains 0), conditioned on different
        values of Y, i.e.:

            c_probs[i*(k+1) + ly, y] = P(\lambda_i = ly | Y = y)

        Note that this simply involves inferring the kth row by law of total
        probability and adding in to mu.

        If `source` is not None, returns only the corresponding block.
        """
        c_probs = np.zeros((self.m * (self.k + 1), self.k))
        mu = self.mu.detach().clone().numpy()

        for i in range(self.m):
            # si = self.c_data[(i,)]['start_index']
            # ei = self.c_data[(i,)]['end_index']
            # mu_i = mu[si:ei, :]
            mu_i = mu[i * self.k : (i + 1) * self.k, :]
            c_probs[i * (self.k + 1) + 1 : (i + 1) * (self.k + 1), :] = mu_i

            # The 0th row (corresponding to abstains) is the difference between
            # the sums of the other rows and one, by law of total prob
            c_probs[i * (self.k + 1), :] = 1 - mu_i.sum(axis=0)
        c_probs = np.clip(c_probs, 0.01, 0.99)

        if source is not None:
            return c_probs[source * (self.k + 1) : (source + 1) * (self.k + 1)]
        else:
            return c_probs

    def get_accuracies(self, probs=None):
        """Returns the vector of LF accuracies, computed using get_conditional_probs"""
        accs = np.zeros(self.m)
        for i in range(self.m):
            if probs is None:
                cps = self.get_conditional_probs(source=i)[1:, :]
            else:
                cps = probs[i * (self.k + 1) : (i + 1) * (self.k + 1)][1:, :]
            accs[i] = np.diag(cps @ self.P.numpy()).sum()
        return accs

    def predict_proba(self, L):
        """Returns the [n,k] matrix of label probabilities P(Y | \lambda)

        Args:
            L: An [n,m] scipy.sparse label matrix with values in {0,1,...,k}
        """
        self._set_constants(L)

        L_aug = self._get_augmented_label_matrix(L)
        mu = np.clip(self.mu.detach().clone().numpy(), 0.01, 0.99)
        jtm = np.ones(L_aug.shape[1])

        # Note: We omit abstains, effectively assuming uniform distribution here
        X = np.exp(L_aug @ np.diag(jtm) @ np.log(mu) + np.log(self.p))
        Z = np.tile(X.sum(axis=1).reshape(-1, 1), self.k)
        return X / Z

    # These loss functions get all their data directly from the LabelModel
    # (for better or worse). The unused *args make these compatible with the
    # Classifer._train() method which expect loss functions to accept an input.

    def loss_l2(self, l2=0):
        """L2 loss centered around mu_init, scaled optionally per-source.

        In other words, diagonal Tikhonov regularization,
            ||D(\mu-\mu_{init})||_2^2
        where D is diagonal.

        Args:
            - l2: A float or np.array representing the per-source regularization
                strengths to use
        """
        if isinstance(l2, (int, float)):
            D = l2 * torch.eye(self.d)
        else:
            D = torch.diag(torch.from_numpy(l2)).type(torch.float32)

        # Note that mu is a matrix and this is the *Frobenius norm*
        return torch.norm(D @ (self.mu - self.mu_init)) ** 2

    def loss_mu(self, *args, l2=0):
        loss_1 = torch.norm((self.O - self.mu @ self.P @ self.mu.t())[self.mask]) ** 2
        loss_2 = torch.norm(torch.sum(self.mu @ self.P, 1) - torch.diag(self.O)) ** 2
        return loss_1 + loss_2 + self.loss_l2(l2=l2)

    def _set_class_balance(self, class_balance, Y_dev):
        """Set a prior for the class balance

        In order of preference:
        1) Use user-provided class_balance
        2) Estimate balance from Y_dev
        3) Assume uniform class distribution
        """
        if class_balance is not None:
            self.p = np.array(class_balance)
        elif Y_dev is not None:
            class_counts = Counter(Y_dev)
            sorted_counts = np.array([v for k, v in sorted(class_counts.items())])
            self.p = sorted_counts / sum(sorted_counts)
        else:
            self.p = (1 / self.k) * np.ones(self.k)
        self.P = torch.diag(torch.from_numpy(self.p)).float()

    def _set_constants(self, L):
        self.n, self.m = L.shape
        self.t = 1

    def _create_tree(self):
        nodes = range(self.m)
        self.c_tree = get_clique_tree(nodes, [])

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
        Y_p, Y, Y_s = map(stack_batches, [Y_p, Y, Y_s])
        if return_probs:
            return Y_p, Y, Y_s
        else:
            return Y_p, Y

    def _execute_logging(self, train_loader, loss, batch_size):
        self.eval()
        self.running_loss += loss.item() * batch_size
        self.running_examples += batch_size

        # Initialize metrics dict
        metrics_dict = {}
        # Always add average loss
        metrics_dict["train/loss"] = self.running_loss / self.running_examples

        if self.logger.check(batch_size):
            logger_metrics = self.logger.calculate_metrics(
                self, train_loader, None, metrics_dict
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

    def _checkpoint(self, metrics_dict):
        if self.checkpointer is None:
            return
        iteration = self.logger.unit_total
        self.checkpointer.checkpoint(
            metrics_dict, iteration, self, self.optimizer, self.lr_scheduler
        )

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

    def _set_optimizer(self, train_config):
        optimizer_config = train_config["optimizer_config"]
        opt = optimizer_config["optimizer"]

        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if opt == "sgd":
            optimizer = optim.SGD(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["sgd_config"],
            )
        elif opt == "rmsprop":
            optimizer = optim.RMSprop(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["rmsprop_config"],
            )
        elif opt == "adam":
            optimizer = optim.Adam(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["adam_config"],
            )
        elif opt == "sparseadam":
            optimizer = optim.SparseAdam(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["adam_config"],
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

    def _train_model(self, train_data, loss_fn):
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
                metrics_dict = self._execute_logging(train_loader, loss, batch_size)
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

    def train_model(
        self, L_train, Y_dev=None, class_balance=None, log_writer=None, **kwargs
    ):
        """Train the model (i.e. estimate mu):

        Args:
            L_train: An [n,m] scipy.sparse matrix with values in {0,1,...,k}
                corresponding to labels from supervision sources on the
                training set
            Y_dev: Target labels for the dev set, for estimating class_balance
            class_balance: (np.array) each class's percentage of the population

        No dependencies (conditionally independent sources): Estimate mu
        subject to constraints:
            (1a) O_{B(i,j)} - (mu P mu.T)_{B(i,j)} = 0, for i != j, where B(i,j)
                is the block of entries corresponding to sources i,j
            (1b) np.sum( mu P, 1 ) = diag(O)
        """
        self.config = recursive_merge_dicts(self.config, kwargs, misses="ignore")
        train_config = self.config["train_config"]

        # TODO: Implement logging for label model?
        if log_writer is not None:
            raise NotImplementedError("Logging for LabelModel.")

        # Note that the LabelModel class implements its own (centered) L2 reg.
        l2 = train_config.get("l2", 0)

        L_train = self._check_L(L_train)
        self._set_class_balance(class_balance, Y_dev)
        self._set_constants(L_train)
        self._create_tree()

        # Creating this faux dataset is necessary for now because the LabelModel
        # loss functions do not accept inputs, but Classifer._train_model()
        # expects training data to feed to the loss functions.
        dataset = MetalDataset([0], [0])
        train_loader = DataLoader(dataset)

        # Compute O and initialize params
        if self.config["verbose"]:  # pragma: no cover
            print("Computing O...")
        self._generate_O(L_train)
        self._init_params()

        # Estimate \mu
        if self.config["verbose"]:  # pragma: no cover
            print("Estimating \mu...")
        self._train_model(train_loader, partial(self.loss_mu, l2=l2))

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
