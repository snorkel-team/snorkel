from typing import List, Optional, Sequence

import numpy as np

from .core import Policy


class MeanFieldPolicy(Policy):
    """Sample sequences of TFs according to a distribution.

    Samples sequences of indices of a specified length from a
    user-provided distribution. A distribution over TFs can be
    learned by a TANDA mean-field model, for example.
    See https://hazyresearch.github.io/snorkel/blog/tanda.html

    Parameters
    ----------
    n_tfs
        Total number of TFs
    sequence_length
        Number of TFs to run on each data point
    p
        Probability distribution from which to sample TF indices.
        Must have length ``n_tfs`` and be a valid distribution.
    """

    def __init__(
        self, n_tfs: int, sequence_length: int = 1, p: Optional[Sequence[float]] = None
    ) -> None:
        self._k = sequence_length
        self._p = p
        super().__init__(n_tfs)

    def generate(self) -> List[int]:
        """Generate a sequence of TF indices by sampling from distribution.

        Returns
        -------
        List[int]
            Indices of TFs to run on data point in order.
        """
        return np.random.choice(self._n, size=self._k, p=self._p).tolist()


class RandomPolicy(MeanFieldPolicy):
    """Naive random augmentation policy.

    Samples sequences of TF indices a specified length at random
    from the total number of TFs. Sampling uniformly at random is
    a common baseline approach to data augmentation.

    Parameters
    ----------
    n_tfs
        Total number of TFs
    sequence_length
        Number of TFs to run on each data point
    """

    def __init__(self, n_tfs: int, sequence_length: int = 1) -> None:
        super().__init__(n_tfs, sequence_length=sequence_length, p=None)
