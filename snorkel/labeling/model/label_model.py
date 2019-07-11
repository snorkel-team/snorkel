from collections import Counter
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import issparse

from snorkel.analysis.utils import set_seed
from snorkel.classification.utils import recursive_merge_dicts
from snorkel.labeling.model.graph_utils import get_clique_tree
from snorkel.labeling.model.lm_defaults import lm_default_config
from snorkel.labeling.model.logger import Logger


class LabelModel(nn.Module):
    """A conditionally independent LabelModel to learn labeling function
    accuracies and assign probabilistic labels

    Args:
        k: (int) the cardinality of the classifier
    """

    def __init__(self, cardinality=2, **kwargs):
        super().__init__()
        self.config = recursive_merge_dicts(lm_default_config, kwargs)
        self.cardinality = cardinality

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

        L_ind = np.zeros((self.n, self.m * self.cardinality))
        for y in range(1, self.cardinality + 1):
            # A[x::y] slices A starting at x at intervals of y
            # e.g., np.arange(9)[0::3] == np.array([0,3,6])
            L_ind[:, (y - 1) :: self.cardinality] = np.where(L == y, 1, 0)
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
                "start_index": i * self.cardinality,
                "end_index": (i + 1) * self.cardinality,
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
                C["start_index"] = members[0] * self.cardinality
                C["end_index"] = (members[0] + 1) * self.cardinality
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

            self.mu[i*self.cardinality + j, y] = P(\lambda_i = j | Y = y)

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
        self.mu_init = torch.zeros(self.d, self.cardinality)
        for i in range(self.m):
            for y in range(self.cardinality):
                idx = i * self.cardinality + y
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
        c_probs = np.zeros((self.m * (self.cardinality + 1), self.cardinality))
        mu = self.mu.detach().clone().numpy()

        for i in range(self.m):
            # si = self.c_data[(i,)]['start_index']
            # ei = self.c_data[(i,)]['end_index']
            # mu_i = mu[si:ei, :]
            mu_i = mu[i * self.cardinality : (i + 1) * self.cardinality, :]
            c_probs[
                i * (self.cardinality + 1) + 1 : (i + 1) * (self.cardinality + 1), :
            ] = mu_i

            # The 0th row (corresponding to abstains) is the difference between
            # the sums of the other rows and one, by law of total prob
            c_probs[i * (self.cardinality + 1), :] = 1 - mu_i.sum(axis=0)
        c_probs = np.clip(c_probs, 0.01, 0.99)

        if source is not None:
            return c_probs[
                source * (self.cardinality + 1) : (source + 1) * (self.cardinality + 1)
            ]
        else:
            return c_probs

    def get_accuracies(self, probs=None):
        """Returns the vector of LF accuracies, computed using get_conditional_probs"""
        accs = np.zeros(self.m)
        for i in range(self.m):
            if probs is None:
                cps = self.get_conditional_probs(source=i)[1:, :]
            else:
                cps = probs[
                    i * (self.cardinality + 1) : (i + 1) * (self.cardinality + 1)
                ][1:, :]
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
        Z = np.tile(X.sum(axis=1).reshape(-1, 1), self.cardinality)
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

    def loss_mu(self, l2=0):
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
            self.p = (1 / self.cardinality) * np.ones(self.cardinality)
        self.P = torch.diag(torch.from_numpy(self.p)).float()

    def _set_constants(self, L):
        self.n, self.m = L.shape
        self.t = 1

    def _create_tree(self):
        nodes = range(self.m)
        self.c_tree = get_clique_tree(nodes, [])

    def _execute_logging(self, loss):
        self.eval()
        self.running_loss += loss.item()
        self.running_examples += 1

        # Always add average loss
        metrics_dict = {"train/loss": self.running_loss / self.running_examples}

        if self.logger.check():
            self.logger.log(metrics_dict)

            # Reset running loss and examples counts
            self.running_loss = 0.0
            self.running_examples = 0

        self.train()
        return metrics_dict

    def _set_logger(self, train_config, epoch_size):
        self.logger = Logger(train_config["log_train_every"])

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

    def train_model(self, L_train, Y_dev=None, class_balance=None, **kwargs):
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

        L_train = self._check_L(L_train)
        self._set_class_balance(class_balance, Y_dev)
        self._set_constants(L_train)
        self._create_tree()

        # Compute O and initialize params
        if self.config["verbose"]:  # pragma: no cover
            print("Computing O...")
        self._generate_O(L_train)
        self._init_params()

        # Estimate \mu
        if self.config["verbose"]:  # pragma: no cover
            print("Estimating \mu...")

        # Set model to train mode
        self.train()
        train_config = self.config["train_config"]
        l2 = train_config.get("l2", 0)

        # Move model to GPU
        if self.config["verbose"] and self.config["device"] != "cpu":
            print("Using GPU...")
        self.to(self.config["device"])

        # Set training components
        self._set_logger(train_config, epoch_size=1)
        self._set_optimizer(train_config)
        self._set_scheduler(train_config)

        # Restore model if necessary
        start_iteration = 0

        # Train the model
        metrics_hist = {}  # The most recently seen value for all metrics
        for epoch in range(start_iteration, train_config["n_epochs"]):
            self.running_loss = 0.0
            self.running_examples = 0

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass to calculate the average loss per example
            loss = self.loss_mu(l2=l2)
            if torch.isnan(loss):
                msg = "Loss is NaN. Consider reducing learning rate."
                raise Exception(msg)

            # Backward pass to calculate gradients
            # Loss is an average loss per example
            loss.backward()

            # Perform optimizer step
            self.optimizer.step()

            # Calculate metrics, log, and checkpoint as necessary
            metrics_dict = self._execute_logging(loss)
            metrics_hist.update(metrics_dict)

            # Apply learning rate scheduler
            self._update_scheduler(epoch, metrics_hist)

        self.eval()

        # Print confusion matrix if applicable
        if self.config["verbose"]:
            print("Finished Training")

    def save(self, destination, **kwargs):
        with open(destination, "wb") as f:
            torch.save(self, f, **kwargs)

    @staticmethod
    def load(source, **kwargs):
        with open(source, "rb") as f:
            return torch.load(f, **kwargs)
