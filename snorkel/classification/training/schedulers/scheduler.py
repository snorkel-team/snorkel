from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Sequence, Tuple

from torch import Tensor

from snorkel.classification.data import DictDataLoader  # noqa: F401

BatchIterator = Iterator[
    Tuple[Tuple[Dict[str, Any], Dict[str, Tensor]], "DictDataLoader"]
]


class Scheduler(ABC):
    """Return batches from all dataloaders according to a specified strategy."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_batches(self, dataloaders: Sequence["DictDataLoader"]) -> BatchIterator:
        """Return batches from dataloaders according to a specified strategy.

        Parameters
        ----------
        dataloaders
            A sequence of dataloaders to get batches from

        Yields
        ------
        (batch, dataloader)
            batch is a tuple of (X_dict, Y_dict) and dataloader is the dataloader
            that that batch came from. That dataloader will not be accessed by the
            model; it is passed primarily so that the model can pull the necessary
            metadata to know what to do with the batch it has been given.
        """
