from typing import List

import numpy as np


class AugmentationPolicy:
    """Base class for augmentation policies.

    Augmentation policies generate sequences of indices, corresponding
    to a sequence of transformation functions to be run on a data point.

    Parameters
    ----------
    n_tfs
        Total number of TFs

    Raises
    ------
    NotImplementedError
        Subclasses need to implement the `generate` method
    """

    def __init__(self, n_tfs: int) -> None:
        self._n = n_tfs

    def generate(self) -> List[int]:
        """Generate a sequence of TF indices.

        Returns
        -------
        List[int]
            Indices of TFs to run on data point in order.

        Raises
        ------
        NotImplementedError
            Subclasses need to implement this method
        """
        raise NotImplementedError


class RandomAugmentationPolicy(AugmentationPolicy):
    """Naive random augmentation policy.

    Samples sequences of TF indices a specified length uniformly
    at random from the total number of TFs. This is a common
    baseline approach to data augmentation.

    Parameters
    ----------
    n_tfs
        Total number of TFs
    sequence_length
        Number of TFs to run on each data point
    """

    def __init__(self, n_tfs: int, sequence_length: int = 1) -> None:
        self._k = sequence_length
        super().__init__(n_tfs)

    def generate(self) -> List[int]:
        """Generate a sequence of TF indices by sampling uniformly at random.

        Returns
        -------
        List[int]
            Indices of TFs to run on data point in order.
        """
        return np.random.choice(self._n, size=self._k).tolist()
