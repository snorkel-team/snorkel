from collections import Counter
from functools import partial
from itertools import chain, product

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse
from torch.utils.data import DataLoader

from snorkel.model.classifier import Classifier
from snorkel.model.utils import MetalDataset, recursive_merge_dicts

from .graph_utils import get_clique_tree
from .lm_defaults import lm_default_config


class LabelModel(Classifier):
    """A conditionally independent LabelModel to learn labeling function accuracies and assign probabilistic labels

    Args:
        k: (int) the cardinality of the classifier
    """

    # This class variable is explained in the Classifier class
    implements_l2 = True

    def __init__(self, k=2, **kwargs):
        config = recursive_merge_dicts(lm_default_config, kwargs)
        super().__init__(k, config)

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
                    C_type = "node"
                elif isinstance(item, tuple):
                    C = self.c_tree[item[0]][item[1]]
                    C_type = "edge"
                else:
                    raise ValueError(item)
                members = list(C["members"])
                nc = len(members)

                # If a unary maximal clique, just store its existing index
                if nc == 1:
                    C["start_index"] = members[0] * self.k
                    C["end_index"] = (members[0] + 1) * self.k

                # Else add one column for each possible value
                else:
                    L_C = np.ones((self.n, self.k ** nc))
                    for i, vals in enumerate(product(range(self.k), repeat=nc)):
                        for j, v in enumerate(vals):
                            L_C[:, i] *= L_ind[:, members[j] * self.k + v]

                    # Add to L_aug and store the indices
                    C["start_index"] = L_aug.shape[1]
                    C["end_index"] = L_aug.shape[1] + L_C.shape[1]
                    L_aug = np.hstack([L_aug, L_C])

                    # Add to self.c_data as well
                    id = tuple(members) if len(members) > 1 else members[0]
                    self.c_data[id] = {
                        "start_index": C["start_index"],
                        "end_index": C["end_index"],
                        "max_cliques": set([item]) if C_type == "node" else set(item),
                    }
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
