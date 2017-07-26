from .constants import *
from numba import jit
import numpy as np
import random


class DependencySelector(object):
    """
    Fast method for identifying dependencies among labeling functions.

    :param seed: seed for initializing state of Numbskull variables
    """
    def __init__(self, seed=271828):
        self.rng = random.Random()
        self.rng.seed(seed)

    def select(self, L, higher_order=False, propensity=False, threshold=0.05, truncation=10):
        """
        Identifies a dependency structure among labeling functions for a given data set.

        By default searches for correlations, i.e., the DEP_SIMILAR dependency type.

        :param L: labeling function output matrix
        :param higher_order: bool indicating whether to additionally search for higher order
                             fixing and reinforcing dependencies (DEP_FIXING and DEP_REINFORCING)
        :param propensity: bool indicating whether to include LF propensity dependencies during learning
        :param threshold: minimum magnitude weight a dependency must have to be returned (in log scale), also
                          regularization strength
        :param truncation: number of iterations between truncation step for regularization
        :return: collection of tuples of the format (LF 1 index, LF 2 index, dependency type),
                 see snorkel.learning.constants
        """
        try:
            L = L.todense()
        except AttributeError:
            pass

        m, n = L.shape

        # Initializes data structures
        deps = set()
        n_weights = 2 * n
        if higher_order:
            n_weights += 4 * n
        if propensity:
            n_weights += 1
        weights = np.zeros((n_weights,))
        joint = np.zeros((6,))
        # joint[0] = P(Y = -1, L_j = -1)
        # joint[1] = P(Y = -1, L_j =  0)
        # joint[2] = P(Y = -1, L_j =  1)
        # joint[3] = P(Y =  1, L_j = -1)
        # joint[4] = P(Y =  1, L_j =  0)
        # joint[5] = P(Y =  1, L_j =  1)

        for j in range(n):
            # Initializes weights
            for k in range(n):
                weights[k] = 1.1 - .2 * self.rng.random()
            for k in range(n, len(weights)):
                weights[k] = 0.0
            if propensity:
                weights[5 * n] = -2.0

            _fit_deps(m, n, j, L, weights, joint, higher_order, propensity,  threshold, truncation)

            for k in range(n):
                if abs(weights[n + k]) > threshold:
                    deps.add((j, k, DEP_SIMILAR) if j < k else (k, j, DEP_SIMILAR))
                if higher_order:
                    if abs(weights[2 * n + k]) > threshold:
                        deps.add((j, k, DEP_REINFORCING))
                    if abs(weights[3 * n + k]) > threshold:
                        deps.add((k, j, DEP_REINFORCING))
                    if abs(weights[4 * n + k]) > threshold:
                        deps.add((j, k, DEP_FIXING))
                    if abs(weights[5 * n + k]) > threshold:
                        deps.add((k, j, DEP_FIXING))

        return deps


@jit(nopython=True, cache=True, nogil=True)
def _fit_deps(m, n, j, L, weights, joint, higher_order, propensity, regularization, truncation):
    step_size = 1.0 / m
    epochs = 10
    l1delta = regularization * step_size * truncation

    for t in range(epochs):
        for i in range(m):
            # Processes a training example

            # First, computes joint and conditional distributions
            joint[:] = 0, 0, 0, 0, 0, 0
            for k in range(n):
                if j == k:
                    # Accuracy
                    joint[0] += weights[j]
                    joint[5] += weights[j]
                    joint[2] -= weights[j]
                    joint[3] -= weights[j]
                else:
                    if L[i, k] == 1:
                        # Accuracy
                        joint[0] -= weights[k]
                        joint[1] -= weights[k]
                        joint[2] -= weights[k]
                        joint[3] += weights[k]
                        joint[4] += weights[k]
                        joint[5] += weights[k]

                        # Similar
                        joint[2] += weights[n + k]
                        joint[5] += weights[n + k]

                        if higher_order:
                            # Reinforcement
                            joint[5] += weights[2 * n + k] + weights[3 * n + k]
                            joint[1] -= weights[2 * n + k]
                            joint[4] -= weights[2 * n + k]

                            # Fixing
                            joint[3] += weights[4 * n + k]
                            joint[1] -= weights[4 * n + k]
                            joint[4] -= weights[4 * n + k]
                            joint[0] += weights[5 * n + k]

                    elif L[i, k] == -1:
                        # Accuracy
                        joint[0] += weights[k]
                        joint[1] += weights[k]
                        joint[2] += weights[k]
                        joint[3] -= weights[k]
                        joint[4] -= weights[k]
                        joint[5] -= weights[k]

                        # Similar
                        joint[0] += weights[n + k]
                        joint[3] += weights[n + k]

                        if higher_order:
                            # Reinforcement
                            joint[0] += weights[2 * n + k] + weights[3 * n + k]
                            joint[1] -= weights[2 * n + k]
                            joint[4] -= weights[2 * n + k]

                            # Fixing
                            joint[2] += weights[4 * n + k]
                            joint[1] -= weights[4 * n + k]
                            joint[4] -= weights[4 * n + k]
                            joint[5] += weights[5 * n + k]

                    else:
                        # Similar
                        joint[1] += weights[n + k]
                        joint[4] += weights[n + k]

                        if higher_order:
                            # Reinforcement
                            joint[0] -= weights[3 * n + k]
                            joint[2] -= weights[3 * n + k]
                            joint[3] -= weights[3 * n + k]
                            joint[5] -= weights[3 * n + k]

                            # Fixing
                            joint[0] -= weights[5 * n + k]
                            joint[2] -= weights[5 * n + k]
                            joint[3] -= weights[5 * n + k]
                            joint[5] -= weights[5 * n + k]

            if propensity:
                joint[0] += weights[6 * n]
                joint[2] += weights[6 * n]
                joint[3] += weights[6 * n]
                joint[5] += weights[6 * n]

            joint = np.exp(joint)
            joint /= np.sum(joint)

            marginal_pos = np.sum(joint[3:6])
            marginal_neg = np.sum(joint[0:3])

            if L[i, j] == 1:
                conditional_pos = joint[5] / (joint[2] + joint[5])
                conditional_neg = joint[2] / (joint[2] + joint[5])
            elif L[i, j] == -1:
                conditional_pos = joint[3] / (joint[0] + joint[3])
                conditional_neg = joint[0] / (joint[0] + joint[3])
            else:
                conditional_pos = joint[4] / (joint[1] + joint[4])
                conditional_neg = joint[1] / (joint[1] + joint[4])

            # Second, takes likelihood gradient step

            for k in range(n):
                if j == k:
                    # Accuracy
                    weights[j] -= step_size * (joint[5] + joint[0] - joint[2] - joint[3])
                    if L[i, j] == 1:
                        weights[j] += step_size * (conditional_pos - conditional_neg)
                    elif L[i, j] == -1:
                        weights[j] += step_size * (conditional_neg - conditional_pos)
                else:
                    if L[i, k] == 1:
                        # Accuracy
                        weights[k] -= step_size * (marginal_pos - marginal_neg - conditional_pos + conditional_neg)

                        # Similar
                        weights[n + k] -= step_size * (joint[2] + joint[5])
                        if L[i, j] == 1:
                            weights[n + k] += step_size

                        if higher_order:
                            # Incoming reinforcement
                            weights[2 * n + k] -= step_size * (joint[5] - joint[1] - joint[4])
                            if L[i, j] == 1:
                                weights[2 * n + k] += step_size * conditional_pos
                            elif L[i, j] == 0:
                                weights[2 * n + k] += step_size * -1

                            # Outgoing reinforcement
                            weights[3 * n + k] -= step_size * joint[5]
                            if L[i, j] == 1:
                                weights[3 * n + k] += step_size * conditional_pos

                            # Incoming fixing
                            weights[4 * n + k] -= step_size * (joint[3] - joint[1] - joint[4])
                            if L[i, j] == -1:
                                weights[4 * n + k] += step_size * conditional_pos
                            elif L[i, j] == 0:
                                weights[4 * n + k] += step_size * -1

                            # Outgoing fixing
                            weights[5 * n + k] -= step_size * joint[0]
                            if L[i, j] == -1:
                                weights[5 * n + k] += step_size * conditional_neg
                    elif L[i, k] == -1:
                        # Accuracy
                        weights[k] -= step_size * (marginal_neg - marginal_pos - conditional_neg + conditional_pos)

                        # Similar
                        weights[n + k] -= step_size * (joint[0] + joint[3])
                        if L[i, j] == -1:
                            weights[n + k] += step_size

                        if higher_order:
                            # Incoming reinforcement
                            weights[2 * n + k] -= step_size * (joint[0] - joint[1] - joint[4])
                            if L[i, j] == -1:
                                weights[2 * n + k] += step_size * conditional_neg
                            elif L[i, j] == 0:
                                weights[2 * n + k] += step_size * -1

                            # Outgoing reinforcement
                            weights[3 * n + k] -= step_size * joint[0]
                            if L[i, j] == -1:
                                weights[3 * n + k] += step_size * conditional_neg

                            # Incoming fixing
                            weights[4 * n + k] -= step_size * (joint[2] - joint[1] - joint[4])
                            if L[i, j] == 1:
                                weights[4 * n + k] += step_size * conditional_neg
                            elif L[i, j] == 0:
                                weights[4 * n + k] += step_size * -1

                            # Outgoing fixing
                            weights[5 * n + k] -= step_size * joint[5]
                            if L[i, j] == 1:
                                weights[5 * n + k] += step_size * conditional_pos
                    else:
                        # Similar
                        weights[n + k] -= step_size * (joint[1] + joint[4])
                        if L[i, j] == 0:
                            weights[n + k] += step_size

                        if higher_order:
                            # No effect of incoming reinforcement

                            # Outgoing reinforcement
                            weights[3 * n + k] -= step_size * (-1 * joint[0] - joint[2] - joint[3] - joint[5])
                            if L[i, j] != 0:
                                weights[3 * n + k] += step_size * -1

                            # No effect of incoming fixing

                            # Outgoing fixing
                            weights[5 * n + k] -= step_size * (-1 * joint[0] - joint[2] - joint[3] - joint[5])
                            if L[i, j] != 0:
                                weights[5 * n + k] += step_size * -1

            if propensity:
                weights[6 * n] -= step_size * (joint[0] + joint[2] + joint[3] + joint[5])
                if L[i, j] != 0:
                    weights[6 * n] += step_size

            # Third, takes regularization gradient step
            if (t * m + i) % truncation == 0:
                for k in range(len(weights)):
                    weights[k] = max(0, weights[k] - l1delta) if weights[k] > 0 else min(0, weights[k] + l1delta)
