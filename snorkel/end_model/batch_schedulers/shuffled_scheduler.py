import random

from .scheduler import Scheduler


class ShuffledScheduler(Scheduler):
    """Return batches from all dataloaders in shuffled order for each epoch"""

    def __init__(self):
        super().__init__()

    def get_batches(self, dataloaders):
        """Return batches in shuffled order.

        TBD
        """
        batch_counts = [len(dl) for dl in dataloaders]
        dataloader_iters = [iter(dl) for dl in dataloaders]

        dataloader_indices = []
        for idx, count in enumerate(batch_counts):
            dataloader_indices.extend([idx] * count)

        random.shuffle(dataloader_indices)

        for index in dataloader_indices:
            yield next(dataloader_iters[index]), dataloaders[index]
