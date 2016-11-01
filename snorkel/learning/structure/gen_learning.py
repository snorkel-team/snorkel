from ..constants import *
from numba import jit
import numpy as np
import random


class DependencySelector(object):
    """
    Heuristic for identifying dependencies among labeling functions.

    :param seed: seed for initializing state of Numbskull variables
    """
    def __init__(self, seed=271828):
        self.rng = random.Random()
        self.rng.seed(seed)

    def select(self, L, threshold=0.1):
        try:
            L = L.todense()
        except AttributeError:
            pass

        m, n = L.shape

        # Initializes data structures
        deps = {}
        weights = np.zeros((3 * n,))
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

            print weights
            _fit_deps(m, n, j, L, weights, joint)
            print weights

            deps[j] = []
            for k in range(n):
                if abs(weights[n + k]) > threshold:
                    deps[j].append((j, k, DEP_REINFORCING))
                if abs(weights[2 * n + k]) > threshold:
                    deps[j].append((k, j, DEP_REINFORCING))

        return deps


@jit(nopython=True, cache=True, nogil=True)
def _fit_deps(m, n, j, L, weights, joint):
    step_size = 1.0 / m
    epochs = 10
    regularization = 0.1
    truncation = 10
    p_truncation = 1.0 / truncation
    l1delta = regularization * step_size * truncation

    for _ in range(epochs):
        for i in range(m):
            # Processes a training example

            # First, computes joint and conditional distributions
            joint[:] = 0, 0, 0, 0, 0, 0
            for k in range(n):
                if j == k:
                    joint[0] += weights[j]
                    joint[5] += weights[j]
                    joint[2] -= weights[j]
                    joint[3] -= weights[j]
                else:
                    if L[i, k] == 1:
                        joint[0] -= weights[k]
                        joint[1] -= weights[k]
                        joint[2] -= weights[k]
                        joint[3] += weights[k]
                        joint[4] += weights[k]
                        joint[5] += weights[k]

                        joint[5] += weights[n + k] + weights[2 * n + k]
                        joint[1] -= weights[n + k]
                        joint[4] -= weights[n + k]

                    elif L[i, k] == -1:
                        joint[0] += weights[k]
                        joint[1] += weights[k]
                        joint[2] += weights[k]
                        joint[3] -= weights[k]
                        joint[4] -= weights[k]
                        joint[5] -= weights[k]

                        joint[0] += weights[n + k] + weights[2 * n + k]
                        joint[1] -= weights[n + k]
                        joint[4] -= weights[n + k]

                    else:
                        joint[0] -= weights[2 * n + k]
                        joint[2] -= weights[2 * n + k]
                        joint[3] -= weights[2 * n + k]
                        joint[5] -= weights[2 * n + k]

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

                        # Incoming reinforcement
                        weights[n + k] -= step_size * (joint[5] - joint[1] - joint[4])
                        if L[i, j] == 1:
                            weights[n + k] += step_size * conditional_pos
                        elif L[i, j] == 0:
                            weights[n + k] += step_size * -1

                        # Outgoing reinforcement
                        weights[2 * n + k] -= step_size * joint[5]
                        if L[i, j] == 1:
                            weights[2 * n + k] += step_size * conditional_pos
                    elif L[i, k] == -1:
                        # Accuracy
                        weights[k] -= step_size * (marginal_neg - marginal_pos - conditional_neg + conditional_pos)

                        # Incoming reinforcement
                        weights[n + k] -= step_size * (joint[0] - joint[1] - joint[4])
                        if L[i, j] == -1:
                            weights[n + k] += step_size * conditional_neg
                        elif L[i, j] == 0:
                            weights[n + k] += step_size * -1

                        # Outgoing reinforcement
                        weights[2 * n + k] -= step_size * joint[0]
                        if L[i, j] == -1:
                            weights[2 * n + k] += step_size * conditional_neg
                    else:
                        # No effect of incoming reinforcement

                        # Outgoing reinforcement
                        weights[2 * n + k] -= step_size * (-1 * joint[0] - joint[2] - joint[3] - joint[5])
                        if L[i, j] != 0:
                            weights[2 * n + k] += step_size * -1

            # Third, takes regularization gradient step
            if random.random() < p_truncation:
                for k in range(3 * n):
                    weights[k] = max(0, weights[k] - l1delta) if weights[k] > 0 else min(0, weights[k] + l1delta)

        loss = 0.0
        for i in range(m):
            joint[:] = 0, 0, 0, 0, 0, 0
            for k in range(n):
                if j == k:
                    joint[0] += weights[j]
                    joint[5] += weights[j]
                    joint[2] -= weights[j]
                    joint[3] -= weights[j]
                else:
                    if L[i, k] == 1:
                        joint[0] -= weights[k]
                        joint[1] -= weights[k]
                        joint[2] -= weights[k]
                        joint[3] += weights[k]
                        joint[4] += weights[k]
                        joint[5] += weights[k]

                        joint[5] += weights[n + k] + weights[2 * n + k]
                        joint[1] -= weights[n + k]
                        joint[4] -= weights[n + k]

                    elif L[i, k] == -1:
                        joint[0] += weights[k]
                        joint[1] += weights[k]
                        joint[2] += weights[k]
                        joint[3] -= weights[k]
                        joint[4] -= weights[k]
                        joint[5] -= weights[k]

                        joint[0] += weights[n + k] + weights[2 * n + k]
                        joint[1] -= weights[n + k]
                        joint[4] -= weights[n + k]

                    else:
                        joint[0] -= weights[2 * n + k]
                        joint[2] -= weights[2 * n + k]
                        joint[3] -= weights[2 * n + k]
                        joint[5] -= weights[2 * n + k]

            joint = np.exp(joint)
            joint /= np.sum(joint)

            if L[i, j] == -1:
                loss -= np.log(joint[0] + joint[3])
            elif L[i, j] == 1:
                loss -= np.log(joint[2] + joint[5])
            else:
                loss -= np.log(joint[1] + joint[4])
        print(loss)

