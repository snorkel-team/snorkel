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
    n_per_original
        Number of transformed data points per original
    keep_original
        Keep untransformed data point in augmented data set? Note that
        even if in-place modifications are made to the original data
        point by the TFs being applied, the original data point will
        remain unchanged.

    Attributes
    ----------
    n
        Total number of TFs
    n_per_original
        See above
    keep_original
        See above
    sequence_length
        See above
    """

    def __init__(
        self,
        n_tfs: int,
        sequence_length: int = 1,
        p: Optional[Sequence[float]] = None,
        n_per_original: int = 1,
        keep_original: bool = True,
    ) -> None:
        self.sequence_length = sequence_length
        self._p = p
        super().__init__(
            n_tfs, n_per_original=n_per_original, keep_original=keep_original
        )

    def generate(self) -> List[int]:
        """Generate a sequence of TF indices by sampling from distribution.

        Returns
        -------
        List[int]
            Indices of TFs to run on data point in order.
        """
        return np.random.choice(self.n, size=self.sequence_length, p=self._p).tolist()


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
    n_per_original
        Number of transformed data points per original
    keep_original
        Keep untransformed data point in augmented data set? Note that
        even if in-place modifications are made to the original data
        point by the TFs being applied, the original data point will
        remain unchanged.

    Attributes
    ----------
    n
        Total number of TFs
    n_per_original
        See above
    keep_original
        See above
    sequence_length
        See above
    """

    def __init__(
        self,
        n_tfs: int,
        sequence_length: int = 1,
        n_per_original: int = 1,
        keep_original: bool = True,
    ) -> None:
        super().__init__(
            n_tfs,
            sequence_length=sequence_length,
            p=None,
            n_per_original=n_per_original,
            keep_original=keep_original,
        )
