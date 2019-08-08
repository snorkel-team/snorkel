from typing import Iterator, List

from tqdm import tqdm

from snorkel.augmentation.policy.core import Policy
from snorkel.augmentation.tf import BaseTransformationFunction
from snorkel.types import DataPoint, DataPoints
from snorkel.utils.data_operators import check_unique_names


class BaseTFApplier:
    """Base class for TF applier objects.

    Base class for TF applier objects, which execute a set of TF
    on a collection of data points. Subclasses should operate on
    a single data point collection format (e.g. ``DataFrame``).
    Subclasses must implement the ``apply`` method.

    Parameters
    ----------
    tfs
        TFs that this applier executes on examples
    policy
        Augmentation policy used to generate sequences of TFs

    Raises
    ------
    ValueError
        If names of TFs are not unique
    """

    def __init__(self, tfs: List[BaseTransformationFunction], policy: Policy) -> None:
        self._tfs = tfs
        self._tf_names = [tf.name for tf in tfs]
        check_unique_names(self._tf_names)
        self._policy = policy

    def _apply_policy_to_data_point(self, x: DataPoint) -> DataPoints:
        x_transformed = []
        for seq in self._policy.generate_for_example():
            x_t = x
            # Handle empty sequence for `keep_original`
            transform_applied = len(seq) == 0
            # Apply TFs
            for tf_idx in seq:
                tf = self._tfs[tf_idx]
                x_t_or_none = tf(x_t)
                # Update if transformation was applied
                if x_t_or_none is not None:
                    transform_applied = True
                    x_t = x_t_or_none
            # Add example if original or transformations applied
            if transform_applied:
                x_transformed.append(x_t)
        return x_transformed

    def __repr__(self) -> str:
        policy_name = type(self._policy).__name__
        return f"{type(self).__name__}, Policy: {policy_name}, TFs: {self._tf_names}"


class TFApplier(BaseTFApplier):
    """TF applier for a list of data points.

    Augments a list of data points (e.g. ``SimpleNamespace``). Primarily
    useful for testing.
    """

    def apply_generator(
        self, data_points: DataPoints, batch_size: int
    ) -> Iterator[List[DataPoint]]:
        """Augment a list of data points using TFs and policy in batches.

        This method acts as a generator, yielding augmented data points for
        a given input batch of data points. This can be useful in a training
        loop when it is too memory-intensive to pregenerate all transformed
        examples.

        Parameters
        ----------
        data_points
            List containing data points to be transformed
        batch_size
            Batch size for generator. Yields augmented data points
            for the next ``batch_size`` input data points.

        Yields
        ------
        List[DataPoint]
            List of data points in augmented data set for batches of inputs
        """
        for i in range(0, len(data_points), batch_size):
            batch_transformed: List[DataPoint] = []
            for x in data_points[i : i + batch_size]:
                batch_transformed.extend(self._apply_policy_to_data_point(x))
            yield batch_transformed

    def apply(
        self, data_points: DataPoints, progress_bar: bool = True
    ) -> List[DataPoint]:
        """Augment a list of data points using TFs and policy.

        Parameters
        ----------
        data_points
            List containing data points to be transformed
        progress_bar
            Display a progress bar?

        Returns
        -------
        List[DataPoint]
            List of data points in augmented data set
        """
        x_transformed: List[DataPoint] = []
        for x in tqdm(data_points, disable=(not progress_bar)):
            x_transformed.extend(self._apply_policy_to_data_point(x))
        return x_transformed
