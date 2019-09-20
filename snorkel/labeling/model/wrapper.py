from typing import Any, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import PredefinedSplit

from snorkel.labeling import LabelModel
from snorkel.utils.lr_schedulers import LRSchedulerConfig
from snorkel.utils.optimizers import OptimizerConfig


class SklearnLabelModel(BaseEstimator, ClassifierMixin):
    """A sklearn wrapper for LabelModel for using sklearn GridSearch, RandomSearch.

    Uses output of create_param_search_data.

    Note that all hyperparameters for the fit and score functions are accepted at the time the class is defined.

    Examples
    --------
    >>> L_train = np.array([[1, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 1]])
    >>> L_dev = np.array([[1, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
    >>> Y_dev = np.array([1, 1, 0, 1])

    >>> from sklearn.model_selection import GridSearchCV
    >>> param_grid = [{"lr": [0.01, 0.0000001],
    ...    "l2": [0.001, 0.5], "metric": ["accuracy"]}]
    >>> label_model = SklearnLabelModel()
    >>> L, Y, cv_split = label_model.create_param_search_data(L_train, L_dev, Y_dev)
    >>> clf = GridSearchCV(label_model, param_grid, cv=cv_split)
    >>> clf = clf.fit(L, Y)
    >>> clf.best_score_
    0.75

    >>> from sklearn.model_selection import RandomizedSearchCV
    >>> param_dist = {"lr": np.random.uniform(low=1e-7, high=1e-2, size=(50,)),
    ...     "l2": np.random.uniform(low=1e-2, high=0.5, size=(50,))}
    >>> clf = RandomizedSearchCV(label_model, param_distributions=param_dist, n_iter=4, cv=cv_split, iid=False)
    >>> clf = clf.fit(L, Y)
    >>> np.around(clf.best_score_)
    1.0

    Parameters
    ----------
    cardinality
         Number of classes, by default 2
    verbose
         Whether to include print statements
    device
        What device to place the model on ('cpu' or 'cuda:0', for example)
    metric
        The metric to report with score()
    tie_break_policy
            Policy to break ties when converting probabilistic labels to predictions
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
    mu_eps
        Restrict the learned conditional probabilities to [mu_eps, 1-mu_eps]
    """

    def __init__(
        self,
        cardinality: int = 2,
        verbose: bool = True,
        device: str = "cpu",
        metric: str = "accuracy",
        tie_break_policy: str = "abstain",
        n_epochs: int = 100,
        lr: float = 0.01,
        l2: float = 0.0,
        optimizer: str = "sgd",
        optimizer_config: Optional[OptimizerConfig] = None,
        lr_scheduler: str = "constant",
        lr_scheduler_config: Optional[LRSchedulerConfig] = None,
        prec_init: float = 0.7,
        seed: int = np.random.randint(1e6),
        log_freq: int = 10,
        mu_eps: Optional[float] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:

        self.cardinality = cardinality
        self.verbose = verbose
        self.device = device
        self.metric = metric
        self.tie_break_policy = tie_break_policy
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2 = l2
        self.optimizer = optimizer
        self.optimizer_config = (
            optimizer_config
            if optimizer_config is not None
            else OptimizerConfig()  # type: ignore
        )
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_config = (
            lr_scheduler_config
            if lr_scheduler_config is not None
            else LRSchedulerConfig()  # type: ignore
        )
        self.prec_init = prec_init
        self.seed = seed
        self.log_freq = log_freq
        self.mu_eps = mu_eps
        self.class_balance = class_balance

        self.label_model = LabelModel(
            cardinality=self.cardinality, verbose=self.verbose, device=self.device
        )

    def fit(self, L: np.ndarray, Y: Optional[np.ndarray] = None) -> "SklearnLabelModel":
        """
        Train label model.

        Parameters
        ----------
        L
             An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y
             Placeholder, not used for training model.

        Returns
        -------
        SklearnLabelModel
        """
        self.label_model.fit(
            L_train=L,
            class_balance=self.class_balance,
            n_epochs=self.n_epochs,
            lr=self.lr,
            l2=self.l2,
            optimizer=self.optimizer,
            optimizer_config=self.optimizer_config,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_config=self.lr_scheduler_config,
            prec_init=self.prec_init,
            seed=self.seed,
            log_freq=self.log_freq,
            mu_eps=self.mu_eps,
        )

        return self

    def score(self, L: np.ndarray, Y: np.ndarray) -> float:
        """Calculate score using self.metric.

        Parameters
        ----------
        L
             An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y
             Gold labels associated with data points in L

        Returns
        -------
        float
            Score for metric speficied in self.metric
        """

        results = self.label_model.score(L, Y, [self.metric], self.tie_break_policy)
        return results[self.metric]

    @staticmethod
    def create_param_search_data(
        L_train: np.ndarray, L_dev: np.ndarray, Y_dev: np.ndarray
    ) -> PredefinedSplit:
        """
        Create predefined cross validation split for SklearnLabelModel wrapper.

        Returns combined L, Y matrix with corresponding CV split object with two splits.

        Parameters
        ----------
        L_train
            An [n,m] matrix with values in {-1,0,1,...,k-1} used for training
        L_dev
            An [n,m] matrix with values in {-1,o0,1,...,k-1} used for scoring
        Y_dev
            Gold labels associated with L_dev

        Returns
        -------
        L
            Combined L_train and L_dev matrix
        Y
            Combined Y_train (all -1s) and Y_dev array
        cv_split
            PredefinedSplit object with two splits: L_train in train and *_dev in test
        """
        n_train = np.shape(L_train)[0]
        n_dev = np.shape(L_dev)[0]

        if n_dev != np.shape(Y_dev)[0]:
            raise ValueError("Num. datapoints in Y_dev and L_dev do not match.")

        # combine train and dev L and Y
        L_all = np.append(L_train, L_dev, axis=0)
        Y_all = np.append(-1 * np.ones(n_train, dtype=int), Y_dev)

        # create cv split array with one predefined split (train, dev)
        test_fold = np.append(
            -1 * np.ones(n_train, dtype=int), np.ones(n_dev, dtype=int)
        )
        cv_split = PredefinedSplit(test_fold=test_fold)
        return L_all, Y_all, cv_split

    def __repr__(self) -> str:
        """Pretty print."""
        return str(vars(self))
