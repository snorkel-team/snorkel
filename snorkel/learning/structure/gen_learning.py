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
        conditional = np.zeros((6,))

        for j in range(n):
            # Initializes weights
            for k in range(n):
                weights[k] = 1.1 - .2 * self.rng.random()
            for k in range(n, len(weights)):
                weights[k] = 0.0

            print weights
            _fit_deps(m, n, j, L, weights, joint, conditional)
            print weights

            deps[j] = []
            for k in range(n):
                if weights[n + k] > threshold:
                    deps[j].append((j, k, DEP_REINFORCING))
                if weights[2 * n + k] > threshold:
                    deps[j].append((k, j, DEP_REINFORCING))
            break

        return deps


@jit(nopython=True, cache=True, nogil=True)
def _fit_deps(m, n, j, L, weights, joint, conditional):
    step_size = 1.0 / m
    epochs = 10

    for _ in range(epochs):
        for i in range(m):
            # Processes a training example

            # First, computes joint and conditional distributions
            joint[:] = 0, 0, 0, 0, 0, 0
            for k in range(n):
                if L[i, k] == 1:
                    joint[0] -= weights[k]
                    joint[1] -= weights[k]
                    joint[2] -= weights[k]
                    joint[3] += weights[k]
                    joint[4] += weights[k]
                    joint[5] += weights[k]

                    if j != k:
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

                    if j != k:
                        joint[0] += weights[n + k] + weights[2 * n + k]
                        joint[1] -= weights[n + k]
                        joint[4] -= weights[n + k]

                elif j != k:
                    joint[1] -= weights[2 * n + k]
                    joint[4] -= weights[2 * n + k]

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

            # Second, takes gradient step

            for k in range(n):
                if j == k:
                    # Accuracy
                    weights[j] -= step_size * (joint[5] + joint[0] - joint[2] - joint[3])
                    if L[i, j] == 1:
                        weights[j] += step_size * (conditional_pos - conditional_neg)
                    elif L[i, j] == -1:
                        weights[j] += step_size * (conditional_neg - conditional_pos)
                else:
                    # Accuracy
                    if L[i, k] == 1:
                        weights[k] -= step_size * (marginal_pos - marginal_neg - conditional_pos + conditional_neg)

                        for i in range(6):
                            print joint[i]
                        print
                        print marginal_pos
                        print marginal_neg
                        print conditional_pos
                        print conditional_neg
                        return
                    elif L[i, k] == -1:
                        weights[k] -= step_size * (marginal_neg - marginal_pos - conditional_neg + conditional_pos)

                    # Incoming reinforcement

                    # Outgoing reinforcement
                    pass




