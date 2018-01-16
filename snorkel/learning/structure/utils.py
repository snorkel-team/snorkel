from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from .constants import *
import random


def get_deps(weights, threshold=0.05, expand=0.0):
    deps = set()
    for dep_mat, dep in (
            (weights.dep_fixing, DEP_FIXING),
            (weights.dep_reinforcing, DEP_REINFORCING),
            (weights.dep_similar, DEP_SIMILAR),
            (weights.dep_exclusive,  DEP_EXCLUSIVE)):
        for i in range(weights.n):
            for j in range(weights.n):
                if abs(dep_mat[i, j]) > threshold or (random.random() < expand and i != j):
                    deps.add((i, j, dep))

    return deps


def get_all_deps(n, dep_fixing=False, dep_reinforcing=False, dep_similar=False, dep_exclusive=False):
    """
    Convenience method for getting a list of all dependencies to consider learning for a given number of labeling
    functions.

    No self dependencies are included, i.e., (i, i, _). In cases of symmetric dependencies, e.g., DEP_SIMILAR, only the
    first case, (i, j, _) where i < j, is included.

    :param n: number of labeling functions
    :param dep_fixing: whether to include DEP_FIXING dependencies. Default is False.
    :param dep_reinforcing: whether to include DEP_REINFORCING dependencies. Default is False.
    :param dep_similar: whether to include DEP_SIMILAR dependencies. Default is False.
    :param dep_exclusive: whether to include DEP_DEP_EXCLUSIVE dependencies. Default is False.
    """
    deps = []

    # Symmetric dependencies
    if dep_similar and dep_exclusive:
        sym_deps = (DEP_SIMILAR, DEP_EXCLUSIVE)
    elif dep_similar:
        sym_deps = (DEP_SIMILAR,)
    elif dep_exclusive:
        sym_deps = (DEP_EXCLUSIVE,)
    else:
        sym_deps = ()

    for dep in sym_deps:
        for i in range(n):
            for j in range(i + 1, n):
                deps.append((i, j, dep))

    # Asymmetric dependencies
    if dep_fixing and dep_reinforcing:
        asym_deps = (DEP_FIXING, DEP_REINFORCING)
    elif dep_fixing:
        asym_deps = (DEP_FIXING,)
    elif dep_reinforcing:
        asym_deps = (DEP_REINFORCING,)
    else:
        asym_deps = ()

    for dep in asym_deps:
        for i in range(n):
            for j in range(n):
                if i != j:
                    deps.append((i, j, dep))

    return deps
