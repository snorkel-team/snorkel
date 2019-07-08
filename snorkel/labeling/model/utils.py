import copy
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class MetalDataset(Dataset):
    """A dataset that group each item in X with its label from Y

    Args:
        X: an n-dim iterable of items
        Y: a torch.Tensor of labels
            This may be predicted (int) labels [n] or probabilistic (float) labels [n, k]
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        assert len(X) == len(Y)

    def __getitem__(self, index):
        return tuple([self.X[index], self.Y[index]])

    def __len__(self):
        return len(self.X)


def recursive_merge_dicts(x, y, misses="report", verbose=None):
    """
    Merge dictionary y into a copy of x, overwriting elements of x when there
    is a conflict, except if the element is a dictionary, in which case recurse.

    misses: what to do if a key in y is not in x
        'insert'    -> set x[key] = value
        'exception' -> raise an exception
        'report'    -> report the name of the missing key
        'ignore'    -> do nothing
    verbose: If verbose is None, look for a value for verbose in y first, then x

    TODO: give example here (pull from tests)
    """

    def recurse(x, y, misses="report", verbose=1):
        found = True
        for k, v in y.items():
            found = False
            if k in x:
                found = True
                if isinstance(x[k], dict):
                    if not isinstance(v, dict):
                        msg = f"Attempted to overwrite dict {k} with " f"non-dict: {v}"
                        raise ValueError(msg)
                    # If v is {}, set x[k] = {} instead of recursing on empty dict
                    # Otherwise, recurse on the items in v
                    if v:
                        recurse(x[k], v, misses, verbose)
                    else:
                        x[k] = v
                else:
                    if x[k] == v:
                        msg = f"Reaffirming {k}={x[k]}"
                    else:
                        msg = f"Overwriting {k}={x[k]} to {k}={v}"
                        x[k] = v
                    if verbose > 1 and k != "verbose":
                        print(msg)
            else:
                for kx, vx in x.items():
                    if isinstance(vx, dict):
                        found = recurse(vx, {k: v}, misses="ignore", verbose=verbose)
                    if found:
                        break
            if not found:
                msg = f'Could not find kwarg "{k}" in destination dict.'
                if misses == "insert":
                    x[k] = v
                    if verbose > 1:
                        print(f"Added {k}={v} from second dict to first")
                elif misses == "exception":
                    raise ValueError(msg)
                elif misses == "report":
                    print(msg)
                else:
                    pass
        return found

    # If verbose is not provided, look for an value in y first, then x
    # (Do this because 'verbose' kwarg is often inside one or both of x and y)
    if verbose is None:
        verbose = y.get("verbose", x.get("verbose", 1))

    z = copy.deepcopy(x)
    recurse(z, y, misses, verbose)
    return z


# DEPRECATION: This is replaced by move_to_device
def place_on_gpu(data):
    """Utility to place data on GPU, where data could be a torch.Tensor, a tuple
    or list of Tensors, or a tuple or list of tuple or lists of Tensors"""
    data_type = type(data)
    if data_type in (list, tuple):
        data = [place_on_gpu(data[i]) for i in range(len(data))]
        data = data_type(data)
        return data
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    else:
        return ValueError(f"Data type {type(data)} not recognized.")


def recursive_transform(x, test_func, transform):
    """Applies a transformation recursively to each member of a dictionary

    Args:
        x: a (possibly nested) dictionary
        test_func: a function that returns whether this element should be transformed
        transform: a function that transforms a value
    """
    for k, v in x.items():
        if test_func(v):
            x[k] = transform(v)
        if isinstance(v, dict):
            recursive_transform(v, test_func, transform)
    return x


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Is this necessary?
        torch.backends.cudnn.enabled = True  # type: ignore
        torch.cuda.manual_seed(seed)  # type: ignore
