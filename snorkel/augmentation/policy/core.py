from typing import List


class Policy:
    """Base class for augmentation policies.

    Augmentation policies generate sequences of indices, corresponding
    to a sequence of transformation functions to be run on a data point.

    Parameters
    ----------
    n_tfs
        Total number of TFs
    n_per_original
        Number of transformed data points for each original data point
    keep_original
        Keep untransformed data point in augmented data set? Note that
        even if in-place modifications are made to the original data
        point by the TFs being applied, the original data point will
        remain unchanged.

    Raises
    ------
    NotImplementedError
        Subclasses need to implement the ``generate`` method

    Attributes
    ----------
    n
        Total number of TFs
    n_per_original
        See above
    keep_original
        See above
    """

    def __init__(
        self, n_tfs: int, n_per_original: int = 1, keep_original: bool = True
    ) -> None:
        self.n = n_tfs
        self.n_per_original = n_per_original
        self.keep_original = keep_original

    def generate_for_example(self) -> List[List[int]]:
        """Generate all sequences of TF indices for a single example.

        Generates `n_per_original` sequences, and adds an empty
        sequence if `keep_original` is True.

        Returns
        -------
        List[List[int]]
            Sequences of indices of TFs to run on data point in order.
        """
        seqs: List[List[int]] = []
        if self.keep_original:
            # Keep original by adding a null transformation sequence
            seqs.append([])
        for _ in range(self.n_per_original):
            seqs.append(self.generate())
        return seqs

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
    random policy is more standard. See ``RandomPolicy``.

    Parameters
    ----------
    n_tfs
        Total number of TFs
    n_per_original
        Number of transformed data points for each original data point
    keep_original
        Keep untransformed data point in augmented data set? Note that
        even if in-place modifications are made to the original data
        point by the TFs being applied, the original data point will
        remain unchanged.

    Example
    -------
    >>> policy = ApplyAllPolicy(3, n_per_original=2, keep_original=False)
    >>> policy.generate_for_example()
    [[0, 1, 2], [0, 1, 2]]

    Attributes
    ----------
    n
        Total number of TFs
    n_per_original
        See above
    keep_original
        See above
    """

    def generate(self) -> List[int]:
        """Generate indices of all TFs in order.

        Returns
        -------
        List[int]
            Indices of all TFs in order.
        """
        return list(range(self.n))


class ApplyOnePolicy(ApplyAllPolicy):
    """Apply a single TF to each data point."""

    def __init__(self, n_per_original: int = 1, keep_original: bool = True) -> None:
        super().__init__(
            n_tfs=1, n_per_original=n_per_original, keep_original=keep_original
        )


class ApplyEachPolicy(Policy):
    """Apply each TF individually to each data point.

    This can be used as a baseline policy when using
    complex transformations which might degenerate if combined.

    Parameters
    ----------
    n_tfs
        Total number of TFs
    keep_original
        Keep untransformed data point in augmented data set? Note that
        even if in-place modifications are made to the original data
        point by the TFs being applied, the original data point will
        remain unchanged.

    Example
    -------
    >>> policy = ApplyEachPolicy(3, keep_original=True)
    >>> policy.generate_for_example()
    [[], [0], [1], [2]]

    Attributes
    ----------
    n
        Total number of TFs
    n_per_original
        Total number of TFs
    keep_original
        See above
    """

    def __init__(self, n_tfs: int, keep_original: bool = True) -> None:
        super().__init__(n_tfs=n_tfs, n_per_original=n_tfs, keep_original=keep_original)

    def generate_for_example(self) -> List[List[int]]:
        """Generate all length-one sequences for a single example.

        Returns
        -------
        List[List[int]]
            Sequences of indices of TFs to run on data point in order.
        """
        seqs = [[i] for i in range(self.n)]
        if self.keep_original:
            # Keep original by adding a null transformation sequence
            seqs.insert(0, [])
        return seqs
