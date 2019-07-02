from typing import List


class Policy:
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


class ApplyAllPolicy(Policy):
    """Apply all TFs in order to each data point.

    While this can be used as a baseline policy, using a
    random policy is more standard. See `RandomPolicy`.

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


class ApplyOnePolicy(ApplyAllPolicy):
    """Apply a single TF to each data point."""

    def __init__(self):
        super().__init__(n_tfs=1)
