import random
from typing import Sequence

from snorkel.classification.data import DictDataLoader

from .scheduler import BatchIterator, Scheduler


class ShuffledScheduler(Scheduler):
    """Return batches from all dataloaders in shuffled order for each epoch."""

    def __init__(self) -> None:
        super().__init__()

    def get_batches(self, dataloaders: Sequence[DictDataLoader]) -> BatchIterator:
        """Return batches in shuffled order from dataloaders.

        Note that this shuffles the batch order, but it does not shuffle the datasets
        themselves; shuffling the datasets is specified in the DataLoaders directly.

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
        batch_counts = [len(dl) for dl in dataloaders]
        dataloader_iters = [iter(dl) for dl in dataloaders]

        dataloader_indices = []
        for idx, count in enumerate(batch_counts):
            dataloader_indices.extend([idx] * count)

        random.shuffle(dataloader_indices)

        for index in dataloader_indices:
            yield next(dataloader_iters[index]), dataloaders[index]
