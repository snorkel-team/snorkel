from ..constants import *


def get_all_deps(n):
    """
    Convenience method for getting a list of all dependencies for a given number of labeling functions.

    No self dependencies are included, i.e., (i, i, _). In cases of symmetric dependencies, e.g., DEP_SIMILAR, only the
    first case, (i, j, _) where i < j, is included.

    :param n: number of labeling functions
    """
    deps = []
    # Symmetric dependencies
    for dep in (DEP_SIMILAR, DEP_EXCLUSIVE):
        for i in range(n):
            for j in range(i + 1, n):
                deps.append((i, j, dep))

    # Asymmetric dependencies
    for dep in (DEP_FIXING, DEP_REINFORCING):
        for i in range(n):
            for j in range(n):
                if i != j:
                    deps.append((i, j, dep))

    return deps
