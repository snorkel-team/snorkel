from typing import List, Optional, Sequence

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


class ApplyAllAugmentationPolicy(AugmentationPolicy):
    """Apply all TFs in order to each data point.

    While this can be used as a baseline policy, using a
    random policy is more standard. See `RandomAugmentationPolicy`.

    Parameters
    ----------
    n_tfs
        Total number of TFs
    """

    def generate(self) -> List[int]:
        """Generate indices of all TFs in order.

        Returns
        -------
        List[int]
            Indices of all TFs in order.
        """
        return list(range(self._n))


class ApplyOneAugmentationPolicy(ApplyAllAugmentationPolicy):
    """Apply a single TF to each data point."""

    def __init__(self):
        super().__init__(n_tfs=1)


class RandomAugmentationPolicy(AugmentationPolicy):
    """Naive random augmentation policy.

    Samples sequences of TF indices a specified length at random
    from the total number of TFs. Sampling uniformly at random is
    a common baseline approach to data augmentation. A distribution
    over TFs can also be specified. This can be learned by a TANDA
    mean-field model, for example.
    See https://hazyresearch.github.io/snorkel/blog/tanda.html

    Parameters
    ----------
    n_tfs
        Total number of TFs
    sequence_length
        Number of TFs to run on each data point
    p
        Probability distribution from which to sample TF indices.
        Must have length `n_tfs` and be a valid distribution.
    """

    def __init__(
        self, n_tfs: int, sequence_length: int = 1, p: Optional[Sequence[float]] = None
    ) -> None:
        self._k = sequence_length
        self._p = p
        super().__init__(n_tfs)

    def generate(self) -> List[int]:
        """Generate a sequence of TF indices by sampling at random.

        Returns
        -------
        List[int]
            Indices of TFs to run on data point in order.
        """
        return np.random.choice(self._n, size=self._k, p=self._p).tolist()
