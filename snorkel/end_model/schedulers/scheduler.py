from abc import ABC, abstractmethod


class Scheduler(ABC):
    """Return batches from all dataloaders in a specified order
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_batches(self, dataloaders):
        """Return batches in a specified order.

        :param dataloaders: a list of dataloaders
        :type dataloaders: list
        """
        pass
