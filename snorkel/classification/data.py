from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .utils import list_to_tensor

XDict = Dict[str, Any]
YDict = Dict[str, Tensor]
Batch = Tuple[XDict, YDict]


class DictDataset(Dataset):
    """A dataset where both the data fields and labels are stored in as dictionaries.

    Parameters
    ----------
    name
        The name of the dataset (e.g., this will be used to report metrics on a
        per-dataset basis)
    split
        The name of the split that the data in this object represents
    X_dict
        A map from field name to values (e.g., {"tokens": ..., "uids": ...})
    Y_dict
        A map from task name to its corresponding set of labels

    Attributes
    ----------
    name
        See above
    split
        See above
    X_dict
        See above
    Y_dict
        See above

    Raises
    ------
    ValueError
        All values in the Y_dict must be of type torch.Tensor
    """

    def __init__(self, name: str, split: str, X_dict: XDict, Y_dict: YDict) -> None:
        self.name = name
        self.split = split
        self.X_dict = X_dict
        self.Y_dict = Y_dict

        for name, label in self.Y_dict.items():
            if not isinstance(label, Tensor):
                raise ValueError(
                    f"Label {name} should be torch.Tensor, not {type(label)}."
                )

    def __getitem__(self, index: int) -> Tuple[XDict, YDict]:
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        return x_dict, y_dict

    def __len__(self) -> int:
        try:
            return len(next(iter(self.Y_dict.values())))  # type: ignore
        except StopIteration:
            return 0


def collate_dicts(batch: List[Batch]) -> Batch:
    """Combine many one-element dicts into a single many-element dict for both X and Y.

    Parameters
    ----------
    batch
        A list of (x_dict, y_dict) where the values of each are a single element

    Returns
    -------
    Batch
        A tuple of X_dict, Y_dict where the values of each are a merged list or tensor
    """
    X_batch: Dict[str, Any] = defaultdict(list)
    Y_batch: Dict[str, Any] = defaultdict(list)

    for x_dict, y_dict in batch:
        for field_name, value in x_dict.items():
            X_batch[field_name].append(value)
        for label_name, value in y_dict.items():
            Y_batch[label_name].append(value)

    for field_name, values in X_batch.items():
        # Only merge list of tensors
        if isinstance(values[0], Tensor):
            X_batch[field_name] = list_to_tensor(values)

    for label_name, values in Y_batch.items():
        Y_batch[label_name] = list_to_tensor(values)

    return dict(X_batch), dict(Y_batch)


class DictDataLoader(DataLoader):
    """A DataLoader that uses the appropriate collate_fn for a `DictDataset`.

    Parameters
    ----------
    dataset
        A dataset to wrap
    collate_fn
        The collate function to use when combining multiple indexed examples for a
        single batch. Usually the default collate_dicts() method should be used, but
        it can be overriden if you want to use different collate logic.
    kwargs
        Keyword arguments to pass on to DataLoader.__init__()
    """

    def __init__(
        self,
        dataset: DictDataset,
        collate_fn: Callable[..., Any] = collate_dicts,
        **kwargs: Any,
    ) -> None:
        assert isinstance(dataset, DictDataset)
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
