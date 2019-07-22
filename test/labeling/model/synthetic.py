from collections import defaultdict

import numpy as np
import torch
from numpy.random import choice, random
from scipy.sparse import csr_matrix

from .words1k import vocab1k

################################################################################
# Helpers
################################################################################


def indpm(x, y):
    """Plus-minus indicator function"""
    return 1 if x == y else -1


################################################################################
# Single-task (Ls and Ys)
################################################################################


class SingleTaskTreeDepsGenerator:
    """Generates a synthetic single-task L and Y matrix with dependencies

    Args:
        n: (int) The number of data points
        m: (int) The number of labeling sources
        k: (int) The cardinality of the classification task
        class_balance: (np.array) each class's percentage of the population
        theta_range: (tuple) The min and max possible values for theta, the
            class conditional accuracy for each labeling source
        edge_prob: edge density in the graph of correlations between sources
        theta_edge_range: The min and max possible values for theta_edge, the
            strength of correlation between correlated sources

    The labeling functions have class-conditional accuracies, and
    class-unconditional pairwise correlations forming a tree-structured graph.

    Note that k = the # of true classes; thus source labels are in {0,1,...,k}
    because they include abstains.
    """

    def __init__(
        self,
        n,
        m,
        k=2,
        class_balance=None,
        theta_range=(0, 1.5),
        edge_prob=0.0,
        theta_edge_range=(-1, 1),
        **kwargs,
    ):
        self.n = n
        self.m = m
        self.cardinality = k

        # Generate correlation structure: edges self.E, parents dict self.parent
        self._generate_edges(edge_prob)

        # Generate class-conditional LF & edge parameters, stored in self.theta
        self._generate_params(theta_range, theta_edge_range)

        # Generate class balance self.p
        if class_balance is None:
            self.p = np.full(k, 1 / k)
        else:
            self.p = class_balance

        # Generate the true labels self.Y and label matrix self.L
        self._generate_label_matrix()

        # Compute the conditional clique probabilities
        self._get_conditional_probs()

        # Correct output type
        self.L = csr_matrix(self.L, dtype=np.int)

    def _generate_edges(self, edge_prob):
        """Generate a random tree-structured dependency graph based on a
        specified edge probability.

        Also create helper data struct mapping child -> parent.
        """
        self.E, self.parent = [], {}
        for i in range(self.m):
            if random() < edge_prob and i > 0:
                p_i = choice(i)
                self.E.append((p_i, i))
                self.parent[i] = p_i

    def _generate_params(self, theta_range, theta_edge_range):
        self.theta = defaultdict(float)
        for i in range(self.m):
            t_min, t_max = min(theta_range), max(theta_range)
            self.theta[i] = (t_max - t_min) * random(self.cardinality + 1) + t_min

        # Choose random weights for the edges
        te_min, te_max = min(theta_edge_range), max(theta_edge_range)
        for (i, j) in self.E:
            w_ij = (te_max - te_min) * random() + te_min
            self.theta[(i, j)] = w_ij
            self.theta[(j, i)] = w_ij

    def _P(self, i, li, j, lj, y):
        return np.exp(
            self.theta[i][y] * indpm(li, y) + self.theta[(i, j)] * indpm(li, lj)
        )

    def P_conditional(self, i, li, j, lj, y):
        """Compute the conditional probability
            P_\theta(li | lj, y)
            =
            Z^{-1} exp(
                theta_{i|y} \indpm{ \lambda_i = Y }
                + \theta_{i,j} \indpm{ \lambda_i = \lambda_j }
            )
        In other words, compute the conditional probability that LF i outputs
        li given that LF j output lj, and Y = y, parameterized by
            - a class-conditional LF accuracy parameter \theta_{i|y}
            - a symmetric LF correlation paramter \theta_{i,j}
        """
        Z = np.sum([self._P(i, _li, j, lj, y) for _li in range(self.cardinality + 1)])
        return self._P(i, li, j, lj, y) / Z

    def _generate_label_matrix(self):
        """Generate an [n,m] label matrix with entries in {0,...,k}"""
        self.L = np.zeros((self.n, self.m))
        self.Y = np.zeros(self.n, dtype=np.int64)
        for i in range(self.n):
            y = choice(self.cardinality, p=self.p) + 1  # Note that y \in {1,...,k}
            self.Y[i] = y
            for j in range(self.m):
                p_j = self.parent.get(j, 0)
                prob_y = self.P_conditional(j, y, p_j, self.L[i, p_j], y)
                prob_0 = self.P_conditional(j, 0, p_j, self.L[i, p_j], y)
                p = np.ones(self.cardinality + 1) * (1 - prob_y - prob_0) / (self.cardinality - 1)
                p[0] = prob_0
                p[y] = prob_y
                self.L[i, j] = choice(self.cardinality + 1, p=p)

    def _get_conditional_probs(self):
        """Compute the true clique conditional probabilities P(\lC | Y) by
        counting given L, Y; we'll use this as ground truth to compare to.

        Note that this generates an attribute, self.c_probs, that has the same
        definition as returned by `LabelModel.get_conditional_probs`.

        TODO: Can compute these exactly if we want to implement that.
        """
        # TODO: Extend to higher-order cliques again
        self.c_probs = np.zeros((self.m * (self.cardinality + 1), self.cardinality))
        for y in range(1, self.cardinality + 1):
            Ly = self.L[self.Y == y]
            for ly in range(self.cardinality + 1):
                self.c_probs[ly :: (self.cardinality + 1), y - 1] = (
                    np.where(Ly == ly, 1, 0).sum(axis=0) / Ly.shape[0]
                )


################################################################################
# Generating Xs and Ds
################################################################################


def gaussian_bags_of_words(Y, vocab=vocab1k, sigma=1, bag_size=[25, 50], **kwargs):
    """
    Generate Gaussian bags of words based on label assignments
    Args:
        Y: np.array of true labels
        sigma: (float) the standard deviation of the Gaussian distributions
        bag_size: (list) the min and max length of bags of words
    Returns:
        X: (Tensor) a tensor of indices representing tokens
        D: (list) a list of sentences (strings)
    The sentences are conditionally independent, given a label.
    Note that technically we use a half-normal distribution here because we
        take the absolute value of the normal distribution.
    """

    def make_distribution(sigma, num_words):
        p = abs(np.random.normal(0, sigma, num_words))
        return p / sum(p)

    num_words = len(vocab)
    word_dists = {y: make_distribution(sigma, num_words) for y in set(Y)}
    bag_sizes = np.random.choice(range(min(bag_size), max(bag_size)), len(Y))

    X = []
    items = []
    for i, (y, length) in enumerate(zip(Y, bag_sizes)):
        x = torch.from_numpy(np.random.choice(num_words, length, p=word_dists[y]))
        X.append(x)
        items.append(" ".join(vocab[j] for j in x))

    return X, items


def bags_to_counts(bags, vocab_size):
    X = torch.zeros(len(bags), vocab_size, dtype=torch.float)
    for i, bag in enumerate(bags):
        for word in bag:
            X[i, word] += 1
    return X
