from collections import Counter
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snorkel.analysis.utils import probs_to_preds, set_seed
from snorkel.classification.scorer import Scorer
from snorkel.labeling.analysis import LFAnalysis
from snorkel.labeling.model.graph_utils import get_clique_tree
from snorkel.labeling.model.logger import Logger
from snorkel.types import Config
from snorkel.utils.config_utils import merge_config
from snorkel.utils.lr_schedulers import LRSchedulerConfig
from snorkel.utils.optimizers import OptimizerConfig

Metrics = Dict[str, float]


class TrainConfig(Config):
    """Settings for the fit() method of LabelModel.

    Parameters
    ----------
    n_epochs
        The number of epochs to train (where each epoch is a single optimization step)
    lr
        Base learning rate (will also be affected by lr_scheduler choice and settings)
    l2
        Centered L2 regularization strength
    optimizer
        Which optimizer to use (one of ["sgd", "adam", "adamax"])
    optimizer_config
        Settings for the optimizer
    lr_scheduler
        Which lr_scheduler to use (one of ["constant", "linear", "exponential", "step"])
    lr_scheduler_config
        Settings for the LRScheduler
    prec_init
        LF precision initializations / priors
    seed
        A random seed to initialize the random number generator with
    log_freq
        Report loss every this many epochs (steps)
    """

    n_epochs: int = 100
    lr: float = 0.01
    l2: float = 0.0
    optimizer: str = "sgd"
    optimizer_config: OptimizerConfig = OptimizerConfig()  # type: ignore
    lr_scheduler: str = "constant"
    lr_scheduler_config: LRSchedulerConfig = LRSchedulerConfig()  # type: ignore
    prec_init: float = 0.7
    seed: int = np.random.randint(1e6)
    log_freq: int = 10


class LabelModelConfig(Config):
    """Settings for the LabelModel initialization.

    Parameters
    ----------
    verbose
        Whether to include print statements
    device
        What device to place the model on ('cpu' or 'cuda:0', for example)
    """

    verbose: bool = True
    device: str = "cpu"


class _CliqueData(NamedTuple):
    start_index: int
    end_index: int
    max_cliques: Set[int]


class LabelModel(nn.Module):
    """A conditionally independent LabelModel to learn LF accuracies and assign training labels.

    Examples
    --------
    >>> label_model = LabelModel()
    >>> label_model = LabelModel(cardinality=3)
    >>> label_model = LabelModel(cardinality=3, device='cpu')
    >>> label_model = LabelModel(cardinality=3)

    Parameters
    ----------
    cardinality
        Number of classes, by default 2
    **kwargs
        Arguments for changing config defaults

    Raises
    ------
    ValueError
        If config device set to cuda but only cpu is available

    Attributes
    ----------
    cardinality
        Number of classes, by default 2
    config
        Training configuration
    seed
        Random seed
    """

    def __init__(self, cardinality: int = 2, **kwargs: Any) -> None:
        super().__init__()
        self.config: LabelModelConfig = LabelModelConfig(**kwargs)
        self.cardinality = cardinality

        # Confirm that cuda is available if config is using CUDA
        if self.config.device != "cpu" and not torch.cuda.is_available():
            raise ValueError("device=cuda but CUDA not available.")

        # By default, put model in eval mode; switch to train mode in training
        self.eval()

    def _create_L_ind(self, L: np.ndarray) -> np.ndarray:
        """Convert a label matrix with labels in 0...k to a one-hot format.

        Parameters
        ----------
        L
            An [n,m] label matrix with values in {0,1,...,k}

        Returns
        -------
        np.ndarray
            An [n,m*k] dense np.ndarray with values in {0,1}
        """
        L_ind = np.zeros((self.n, self.m * self.cardinality))
        for y in range(1, self.cardinality + 1):
            # A[x::y] slices A starting at x at intervals of y
            # e.g., np.arange(9)[0::3] == np.array([0,3,6])
            L_ind[:, (y - 1) :: self.cardinality] = np.where(L == y, 1, 0)
        return L_ind

    def _get_augmented_label_matrix(
        self, L: np.ndarray, higher_order: bool = False
    ) -> np.ndarray:
        """Create augmented version of label matrix.

        In augmented version, each column is an indicator
        for whether a certain source or clique of sources voted in a certain
        pattern.

        Parameters
        ----------
        L
            An [n,m] label matrix with values in {0,1,...,k}
        higher_order
            Whether to include higher-order correlations (e.g. LF pairs) in matrix

        Returns
        -------
        np.ndarray
            An [n,m*k] dense matrix with values in {0,1}
        """
        # Create a helper data structure which maps cliques (as tuples of member
        # sources) --> {start_index, end_index, maximal_cliques}, where
        # the last value is a set of indices in this data structure
        self.c_data: Dict[int, _CliqueData] = {}
        for i in range(self.m):
            self.c_data[i] = _CliqueData(
                start_index=i * self.cardinality,
                end_index=(i + 1) * self.cardinality,
                max_cliques=set(
                    [
                        j
                        for j in self.c_tree.nodes()
                        if i in self.c_tree.node[j]["members"]
                    ]
                ),
            )

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

    def _build_mask(self) -> None:
        """Build mask applied to O^{-1}, O for the matrix approx constraint."""
        self.mask = torch.ones(self.d, self.d).byte()
        for ci in self.c_data.values():
            si = ci.start_index
            ei = ci.end_index
            for cj in self.c_data.values():
                sj, ej = cj.start_index, cj.end_index

                # Check if ci and cj are part of the same maximal clique
                # If so, mask out their corresponding blocks in O^{-1}
                if len(ci.max_cliques.intersection(cj.max_cliques)) > 0:
                    self.mask[si:ei, sj:ej] = 0
                    self.mask[sj:ej, si:ei] = 0

    def _generate_O(self, L: np.ndarray, higher_order: bool = False) -> None:
        """Generate overlaps and conflicts matrix from label matrix.

        Parameters
        ----------
        L
            An [n,m] label matrix with values in {0,1,...,k}
        higher_order
            Whether to include higher-order correlations (e.g. LF pairs) in matrix
        """
        L_aug = self._get_augmented_label_matrix(L, higher_order=higher_order)
        self.d = L_aug.shape[1]
        self.O = torch.from_numpy(L_aug.T @ L_aug / self.n).float()

    def _init_params(self) -> None:
        r"""Initialize the learned params.

        - \mu is the primary learned parameter, where each row corresponds to
        the probability of a clique C emitting a specific combination of labels,
        conditioned on different values of Y (for each column); that is:

            self.mu[i*self.cardinality + j, y] = P(\lambda_i = j | Y = y)

        and similarly for higher-order cliques.

        Raises
        ------
        ValueError
            If prec_init shape does not match number of LFs
        """
        # Initialize mu so as to break basic reflective symmetry
        # Note that we are given either a single or per-LF initial precision
        # value, prec_i = P(Y=y|\lf=y), and use:
        #   mu_init = P(\lf=y|Y=y) = P(\lf=y) * prec_i / P(Y=y)

        # Handle single values
        if isinstance(self.train_config.prec_init, (int, float)):
            self._prec_init = self.train_config.prec_init * torch.ones(self.m)
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

    def _get_conditional_probs(self, source: Optional[int] = None) -> np.ndarray:
        r"""Return the full conditional probabilities table.

        In cond. prob. table, row i*(k+1) + ly is the conditional probabilities of source i
        emmiting label ly (including abstains 0), conditioned on different
        values of Y, i.e.:

            c_probs[i*(k+1) + ly, y] = P(\lambda_i = ly | Y = y)

        Note that this simply involves inferring the kth row by law of total
        probability and adding in to mu.

        If ``source`` is not None, returns only the corresponding block.

        Parameters
        ----------
        source
            Index of source to generate conditional probabilities for, by default None

        Returns
        -------
        np.ndarray
            Conditional probabilities table if source is None, else corresponding block
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

    def get_accuracies(self) -> np.ndarray:
        """Return the vector of LF accuracies.

        Returns
        -------
        np.ndarray
            [m,1] vector of LF accuracies

        Example
        -------
        >>> L = np.array([[1, 1, 1], [1, 1, -1], [-1, 0, 0], [0, 0, 0]])
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L)
        >>> np.around(label_model.get_accuracies(), 2)
        array([0.99, 0.99, 0.99])
        """
        accs = np.zeros(self.m)
        for i in range(self.m):
            cps = self._get_conditional_probs(source=i)[1:, :]
            accs[i] = np.diag(cps @ self.P.numpy()).sum()

        return np.clip(accs / self.coverage, 1e-6, 1.0)

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        r"""Return label probabilities P(Y | \lambda).

        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}

        Returns
        -------
        np.ndarray
            An [n,k] array of probabilistic labels

        Example
        -------
        >>> L = np.array([[0, 0, -1], [1, 1, -1], [0, 0, -1]])
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L)
        >>> np.around(label_model.predict_proba(L), 1)
        array([[1., 0.],
               [0., 1.],
               [1., 0.]])
        """
        L_shift = L + 1  # convert to {0, 1, ..., k}
        self._set_constants(L_shift)
        L_aug = self._get_augmented_label_matrix(L_shift)
        mu = np.clip(self.mu.detach().clone().numpy(), 0.01, 0.99)
        jtm = np.ones(L_aug.shape[1])

        # Note: We omit abstains, effectively assuming uniform distribution here
        X = np.exp(L_aug @ np.diag(jtm) @ np.log(mu) + np.log(self.p))
        Z = np.tile(X.sum(axis=1).reshape(-1, 1), self.cardinality)
        return X / Z

    def predict(
        self,
        L: np.ndarray,
        return_probs: Optional[bool] = False,
        tie_break_policy: str = "random",
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return predicted labels, with ties broken according to policy.

        Policies to break ties include:
        "abstain": return an abstain vote (0)
        "true-random": randomly choose among the tied options
        "random": randomly choose among tied option using deterministic hash

        NOTE: if tie_break_policy="true-random", repeated runs may have slightly different
        results due to difference in broken ties


        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        return_probs
            Whether to return probs along with preds
        tie_break_policy
            Policy to break ties when converting probabilistic labels to predictions

        Returns
        -------
        np.ndarray
            An [n,1] array of integer labels

        (np.ndarray, np.ndarray)
            An [n,1] array of integer labels and an [n,k] array of probabilistic labels


        Example
        -------
        >>> L = np.array([[0, 0, -1], [1, 1, -1], [0, 0, -1]])
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L)
        >>> label_model.predict(L)
        array([0, 1, 0])
        """
        Y_probs = self.predict_proba(L)
        Y_p = probs_to_preds(Y_probs, tie_break_policy)
        if return_probs:
            return Y_p, Y_probs
        return Y_p

    def score(
        self,
        L: np.ndarray,
        Y: np.ndarray,
        metrics: Optional[List[str]] = ["accuracy"],
        tie_break_policy: str = "random",
    ) -> Dict[str, float]:
        """Calculate one or more scores from user-specified and/or user-defined metrics.

        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y
            Gold labels associated with datapoints in L
        metrics
            A list of metric names
        tie_break_policy
            Policy to break ties when converting probabilistic labels to predictions


        Returns
        -------
        Dict[str, float]
            A dictionary mapping metric names to metric scores

        Example
        -------
        >>> L = np.array([[1, 1, -1], [0, 0, -1], [1, 1, -1]])
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L)
        >>> label_model.score(L, Y=np.array([1, 1, 1]))
        {'accuracy': 0.6666666666666666}
        >>> label_model.score(L, Y=np.array([1, 1, 1]), metrics=["f1"])
        {'f1': 0.8}
        """
        Y_pred, Y_prob = self.predict(
            L, return_probs=True, tie_break_policy=tie_break_policy
        )

        scorer = Scorer(metrics=metrics)
        results = scorer.score(Y, Y_pred, Y_prob)
        return results

    # These loss functions get all their data directly from the LabelModel
    # (for better or worse). The unused *args make these compatible with the
    # Classifer._train() method which expect loss functions to accept an input.
    def _loss_l2(self, l2: float = 0) -> torch.Tensor:
        r"""L2 loss centered around mu_init, scaled optionally per-source.

        In other words, diagonal Tikhonov regularization,
            ||D(\mu-\mu_{init})||_2^2
        where D is diagonal.

        Parameters
        ----------
        l2
            A float or np.array representing the per-source regularization
            strengths to use, by default 0

        Returns
        -------
        torch.Tensor
            L2 loss between learned mu and initial mu
        """
        if isinstance(l2, (int, float)):
            D = l2 * torch.eye(self.d)
        else:
            D = torch.diag(torch.from_numpy(l2)).type(torch.float32)

        # Note that mu is a matrix and this is the *Frobenius norm*
        return torch.norm(D @ (self.mu - self.mu_init)) ** 2

    def _loss_mu(self, l2: float = 0) -> torch.Tensor:
        r"""Overall mu loss.

        Parameters
        ----------
        l2
            A float or np.array representing the per-source regularization
                strengths to use, by default 0

        Returns
        -------
        torch.Tensor
            Overall mu loss between learned mu and initial mu
        """
        loss_1 = torch.norm((self.O - self.mu @ self.P @ self.mu.t())[self.mask]) ** 2
        loss_2 = torch.norm(torch.sum(self.mu @ self.P, 1) - torch.diag(self.O)) ** 2
        return loss_1 + loss_2 + self._loss_l2(l2=l2)

    def _set_class_balance(
        self, class_balance: Optional[List[float]], Y_dev: np.ndarray
    ) -> None:
        """Set a prior for the class balance.

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

    def _set_constants(self, L: np.ndarray) -> None:
        self.n, self.m = L.shape
        self.t = 1

    def _create_tree(self) -> None:
        nodes = range(self.m)
        self.c_tree = get_clique_tree(nodes, [])

    def _execute_logging(self, loss: torch.Tensor) -> Metrics:
        self.eval()
        self.running_examples: int
        self.running_loss: float
        self.running_loss += loss.item()
        self.running_examples += 1

        # Always add average loss
        metrics_dict = {"train/loss": self.running_loss / self.running_examples}

        if self.logger.check():
            if self.config.verbose:
                self.logger.log(metrics_dict)

            # Reset running loss and examples counts
            self.running_loss = 0.0
            self.running_examples = 0

        self.train()
        return metrics_dict

    def _set_logger(self) -> None:
        self.logger = Logger(self.train_config.log_freq)

    def _set_optimizer(self) -> None:
        parameters = filter(lambda p: p.requires_grad, self.parameters())

        optimizer_config = self.train_config.optimizer_config
        optimizer_name = self.train_config.optimizer
        optimizer: optim.Optimizer  # type: ignore

        if optimizer_name == "sgd":
            optimizer = optim.SGD(  # type: ignore
                parameters,
                lr=self.train_config.lr,
                weight_decay=self.train_config.l2,
                **optimizer_config.sgd_config._asdict(),
            )
        elif optimizer_name == "adam":
            optimizer = optim.Adam(
                parameters,
                lr=self.train_config.lr,
                weight_decay=self.train_config.l2,
                **optimizer_config.adam_config._asdict(),
            )
        elif optimizer_name == "adamax":
            optimizer = optim.Adamax(  # type: ignore
                parameters,
                lr=self.train_config.lr,
                weight_decay=self.train_config.l2,
                **optimizer_config.adamax_config._asdict(),
            )
        else:
            raise ValueError(f"Unrecognized optimizer option '{optimizer_name}'")

        self.optimizer = optimizer

    def _set_lr_scheduler(self) -> None:
        # Set warmup scheduler
        self._set_warmup_scheduler()

        # Set lr scheduler
        lr_scheduler_name = self.train_config.lr_scheduler
        lr_scheduler_config = self.train_config.lr_scheduler_config
        lr_scheduler: Optional[optim.lr_scheduler._LRScheduler]

        if lr_scheduler_name == "constant":
            lr_scheduler = None
        elif lr_scheduler_name == "linear":
            total_steps = self.train_config.n_epochs
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

        if self.train_config.lr_scheduler_config.warmup_steps:
            warmup_steps = self.train_config.lr_scheduler_config.warmup_steps
            if warmup_steps < 0:
                raise ValueError(f"warmup_steps much greater or equal than 0.")
            warmup_unit = self.train_config.lr_scheduler_config.warmup_unit
            if warmup_unit == "epochs":
                self.warmup_steps = int(warmup_steps)
            else:
                raise ValueError(
                    "LabelModel does not support any warmup_unit other than 'epochs'."
                )

            linear_warmup_func = lambda x: x / self.warmup_steps
            warmup_scheduler = optim.lr_scheduler.LambdaLR(  # type: ignore
                self.optimizer, linear_warmup_func
            )
            if self.config.verbose:  # pragma: no cover
                print(f"Warmup {self.warmup_steps} steps.")

        elif self.train_config.lr_scheduler_config.warmup_percentage:
            warmup_percentage = self.train_config.lr_scheduler_config.warmup_percentage
            self.warmup_steps = int(warmup_percentage * self.train_config.n_epochs)
            linear_warmup_func = lambda x: x / self.warmup_steps
            warmup_scheduler = optim.lr_scheduler.LambdaLR(  # type: ignore
                self.optimizer, linear_warmup_func
            )
            if self.config.verbose:  # pragma: no cover
                print(f"Warmup {self.warmup_steps} steps.")

        else:
            warmup_scheduler = None
            self.warmup_steps = 0

        self.warmup_scheduler = warmup_scheduler

    def _update_lr_scheduler(self, step: int) -> None:
        if self.warmup_scheduler and step < self.warmup_steps:
            self.warmup_scheduler.step()  # type: ignore
        elif self.lr_scheduler is not None:
            self.lr_scheduler.step()  # type: ignore
            min_lr = self.train_config.lr_scheduler_config.min_lr
            if min_lr and self.optimizer.param_groups[0]["lr"] < min_lr:
                self.optimizer.param_groups[0]["lr"] = min_lr

    def fit(
        self,
        L_train: np.ndarray,
        Y_dev: Optional[np.ndarray] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Train label model.

        Train label model to estimate mu, the parameters related to accuracies of LFs.

        Parameters
        ----------
        L_train
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y_dev
            Gold labels for dev set for estimating class_balance, by default None
        class_balance
            Each class's percentage of the population, by default None
        **kwargs
            Arguments for changing train config defaults

        Raises
        ------
        Exception
            If loss in NaN

        Examples
        --------
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> Y_dev = [0, 1, 0]
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L)
        >>> label_model.fit(L, Y_dev=Y_dev)
        >>> label_model.fit(L, class_balance=[0.7, 0.3])
        """
        # Set random seed
        self.train_config: TrainConfig = merge_config(  # type:ignore
            TrainConfig(), kwargs  # type:ignore
        )
        # Update base config so that it includes all parameters
        set_seed(self.train_config.seed)

        L_shift = L_train + 1  # convert to {0, 1, ..., k}
        self._set_class_balance(class_balance, Y_dev)
        self._set_constants(L_shift)
        self._create_tree()
        lf_analysis = LFAnalysis(L_train)
        self.coverage = lf_analysis.lf_coverages()

        # Compute O and initialize params
        if self.config.verbose:  # pragma: no cover
            print("Computing O...")
        self._generate_O(L_shift)
        self._init_params()

        # Estimate \mu
        if self.config.verbose:  # pragma: no cover
            print("Estimating \mu...")

        # Set model to train mode
        self.train()

        # Move model to GPU
        if self.config.verbose and self.config.device != "cpu":  # pragma: no cover
            print("Using GPU...")
        self.to(self.config.device)

        # Set training components
        self._set_logger()
        self._set_optimizer()
        self._set_lr_scheduler()

        # Restore model if necessary
        start_iteration = 0

        # Train the model
        metrics_hist = {}  # The most recently seen value for all metrics
        for epoch in range(start_iteration, self.train_config.n_epochs):
            self.running_loss = 0.0
            self.running_examples = 0

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass to calculate the average loss per example
            loss = self._loss_mu(l2=self.train_config.l2)
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

            # Update learning rate
            self._update_lr_scheduler(epoch)

        self.eval()

        # Print confusion matrix if applicable
        if self.config.verbose:  # pragma: no cover
            print("Finished Training")

    def save(self, destination: str, **kwargs: Any) -> None:
        """Save label model.

        Parameters
        ----------
        destination
            File location for saving model
        **kwargs
            Arguments for torch.save

        Example
        -------
        >>> label_model.save('./saved_label_model')  # doctest: +SKIP
        """
        with open(destination, "wb") as f:
            torch.save(self, f, **kwargs)

    @staticmethod
    def load(source: str, **kwargs: Any) -> Any:
        """Load existing label model.

        Parameters
        ----------
        source
            File location from where to load model
        **kwargs
            Arguments for torch.load

        Returns
        -------
        LabelModel
            LabelModel with appropriate loaded parameters

        Example
        -------
        Load parameters saved in ``saved_label_model``

        >>> label_model.load('./saved_label_model')  # doctest: +SKIP
        """
        with open(source, "rb") as f:
            return torch.load(f, **kwargs)
