from .scheduler import Scheduler


class SequentialScheduler(Scheduler):
    """Return batches from all dataloaders in sequential order."""

    def __init__(self):
        super().__init__()

    def get_batches(self, dataloaders):
        """Return batches in sequential order.

        TBD
        """
        for dataloader in dataloaders:
            for batch in dataloader:
                yield batch, dataloader
