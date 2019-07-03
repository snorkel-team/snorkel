from typing import Any, Dict, Sequence, Tuple

from snorkel.classification.data import MultitaskDataLoader

from .scheduler import Scheduler


class SequentialScheduler(Scheduler):
    """Return batches from all dataloaders in sequential order."""

    def __init__(self):
        super().__init__()

    def get_batches(
        self, dataloaders: Sequence[MultitaskDataLoader]
    ) -> Tuple[Tuple[Dict[str, Any], Dict[str, Any]], MultitaskDataLoader]:
        """Return batches from dataloaders sequentially in the order they were given.

        Parameters
        ----------
        dataloaders
            A sequence of dataloaders to get batches from

        Yields
        -------
        (batch, dataloader)
            batch is a tuple of (X_dict, Y_dict) and dataloader is the dataloader
            that that batch came from. That dataloader will not be accessed by the
            model; it is passed primarily so that the model can pull the necessary
            metadata to know what to do with the batch it has been given.
        """
        for dataloader in dataloaders:
            for batch in dataloader:
                yield batch, dataloader
