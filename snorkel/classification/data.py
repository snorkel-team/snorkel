import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import list_to_tensor


class MultitaskDataset(Dataset):
    """An advanced dataset class to handle input data with multipled fields and output
    data with multiple label sets

    :param name: the name of the dataset
    :type name: str
    :param X_dict: the feature dict where key is the feature name and value is the
    feature
    :type X_dict: dict
    :param Y_dict: the label dict where key is the label name and value is
    the label
    :type Y_dict: dict
    """

    def __init__(self, name, split, X_dict, Y_dict):
        self.name = name
        self.split = split
        self.X_dict = X_dict
        self.Y_dict = Y_dict

        for name, label in self.Y_dict.items():
            if not isinstance(label, torch.Tensor):
                raise ValueError(
                    f"Label {name} should be torch.Tensor, not {type(label)}."
                )

    def __getitem__(self, index):
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        return x_dict, y_dict

    def __len__(self):
        try:
            return len(next(iter(self.Y_dict.values())))
        except StopIteration:
            return 0


def collate_dicts(batch):

    X_batch = defaultdict(list)
    Y_batch = defaultdict(list)

    for x_dict, y_dict in batch:
        for field_name, value in x_dict.items():
            X_batch[field_name].append(value)
        for label_name, value in y_dict.items():
            Y_batch[label_name].append(value)

    for field_name, values in X_batch.items():
        # Only merge list of tensors
        if isinstance(values[0], torch.Tensor):
            X_batch[field_name] = list_to_tensor(values)

    for label_name, values in Y_batch.items():
        Y_batch[label_name] = list_to_tensor(values)

    return dict(X_batch), dict(Y_batch)


class MultitaskDataLoader(DataLoader):
    """An advanced dataloader class which contains mapping from task to label (which
    label(s) to use in dataset's Y_dict for this task), and split (which part this
    dataset belongs to) information.

    :param task_to_label_dict: the task to label mapping where key is the task name and
    value is the labels for that task and should be the key in Y_dict
    :type task_to_label_dict: dict
    :param dataset: the dataset to construct the dataloader
    :type dataset: torch.utils.data.Dataset
    :param split: the split information, defaults to "train"
    :param split: str, optional
    :param collate_fn: the function that merges a list of samples to form a
    mini-batch, defaults to collate_dicts
    :param collate_fn: function, optional
    """

    def __init__(
        self, dataset, collate_fn=collate_dicts, task_to_label_dict=None, **kwargs
    ):

        assert isinstance(dataset, MultitaskDataset)
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)

        self.task_to_label_dict = task_to_label_dict or {}

        for label in self.task_to_label_dict.values():
            if label not in dataset.Y_dict:
                raise ValueError(
                    f"Label {label} specified in task_to_label_dict could not be found in Y_dict"
                )


def split_data(
    *inputs,
    splits=[0.5, 0.5],
    shuffle=True,
    stratify_by=None,
    index_only=False,
    seed=None,
):
    """Splits inputs into multiple splits of defined sizes

    Args:
        inputs: correlated tuples/lists/arrays/matrices/tensors to split
        splits: list containing split sizes (fractions or counts);
        shuffle: if True, shuffle the data before splitting
        stratify_by: (None or an input) if not None, use these labels to
            stratify the splits (separating the data into groups by these
            labels and sampling from those, rather than from the population at
            large); overrides shuffle
        index_only: if True, return only the indices of the new splits, not the
            split data itself
        seed: (int) random seed

    Example usage:
        Ls, Xs, Ys = split_data(L, X, Y, splits=[0.8, 0.1, 0.1])
        OR
        assignments = split_data(Y, splits=[0.8, 0.1, 0.1], index_only=True)

    Note: This is very similar to scikit-learn's train_test_split() method,
        but with support for more than two splits.
    """

    def fractions_to_counts(fracs, n):
        """Converts a list of fractions to a list of counts that sum to n"""
        counts = [int(np.round(n * frac)) for frac in fracs]
        # Ensure sum of split counts sums to n
        counts[-1] = n - sum(counts[:-1])
        return counts

    def slice_data(data, indices):
        if isinstance(data, list) or isinstance(data, tuple):
            return [d for i, d in enumerate(data) if i in set(indices)]
        else:
            try:
                # Works for np.ndarray, scipy.sparse, torch.Tensor
                return data[indices]
            except TypeError:
                raise Exception(
                    f"split_data() currently only accepts inputs "
                    f"of type tuple, list, np.ndarray, scipy.sparse, or "
                    f"torch.Tensor; not {type(data)}"
                )

    # Setting random seed
    if seed is not None:
        random.seed(seed)

    try:
        n = len(inputs[0])
    except TypeError:
        n = inputs[0].shape[0]
    num_splits = len(splits)

    # Check splits for validity and convert to fractions
    if all(isinstance(x, int) for x in splits):
        if not sum(splits) == n:
            raise ValueError(
                f"Provided split counts must sum to n ({n}), not {sum(splits)}."
            )
        fracs = [count / n for count in splits]

    elif all(isinstance(x, float) for x in splits):
        if not sum(splits) == 1.0:
            raise ValueError(f"Split fractions must sum to 1.0, not {sum(splits)}.")
        fracs = splits

    else:
        raise ValueError("Splits must contain all ints or all floats.")

    # Make sampling pools
    if stratify_by is None:
        pools = [np.arange(n)]
    else:
        pools = defaultdict(list)
        for i, val in enumerate(stratify_by):
            pools[val].append(i)
        pools = list(pools.values())

    # Make index assignments
    assignments = [[] for _ in range(num_splits)]
    for pool in pools:
        if shuffle or stratify_by is not None:
            random.shuffle(pool)

        counts = fractions_to_counts(fracs, len(pool))
        counts.insert(0, 0)
        cum_counts = np.cumsum(counts)
        for i in range(num_splits):
            assignments[i].extend(pool[cum_counts[i] : cum_counts[i + 1]])

    if index_only:
        return assignments
    else:
        outputs = []
        for data in inputs:
            data_splits = []
            for split in range(num_splits):
                data_splits.append(slice_data(data, assignments[split]))
            outputs.append(data_splits)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
