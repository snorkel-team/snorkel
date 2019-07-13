from typing import Any, Iterator, List

from tqdm import tqdm

from snorkel.augmentation.policy import Policy
from snorkel.augmentation.tf import BaseTransformationFunction
from snorkel.types import DataPoint, DataPoints


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
    k
        Number of transformed data points per original
    keep_original
        Keep untransformed data point in augmented data set? Note that
        even if in-place modifications are made to the original data
        point by the TFs being applied, the original data point will
        remain unchanged.

    Raises
    ------
    NotImplementedError
        ``apply`` method must be implemented by subclasses
    """

    def __init__(
        self,
        tfs: List[BaseTransformationFunction],
        policy: Policy,
        k: int = 1,
        keep_original: bool = True,
    ) -> None:
        self._tfs = tfs
        self._policy = policy
        self._k = k
        self._keep_original = keep_original

    def _apply_policy_to_data_point(self, x: DataPoint) -> DataPoints:
        x_transformed = []
        if self._keep_original:
            x_transformed.append(x)
        for _ in range(self._k):
            x_t = x
            transform_applied = False
            for tf_idx in self._policy.generate():
                tf = self._tfs[tf_idx]
                x_t_raw = tf(x_t)
                if x_t_raw is not None:
                    transform_applied = True
                    x_t = x_t_raw
            if transform_applied:
                x_transformed.append(x_t)
        return x_transformed

    def apply(self, data_points: Any, *args: Any, **kwargs: Any) -> Any:
        """Transform a collection of data points with TFs and policy.

        Parameters
        ----------
        data_points
            Collection of data points to be transformed by TFs and policy. Subclasses
            implement functionality for a specific format (e.g. ``DataFrame``).

        Returns
        -------
        Any
            Collection of data points in augmented data set

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError

    def apply_generator(
        self, data_points: Any, batch_size: int, *args: Any, **kwargs: Any
    ) -> Any:
        """Transform a collection of data points with TFs and policy in batches.

        This method acts as a generator, yielding augmented data points for
        a given input batch of data points. This can be useful in a training
        loop when it is too memory-intensive to pregenerate all transformed
        examples.

        Parameters
        ----------
        data_points
            Collection of data points to be transformed by TFs and policy. Subclasses
            implement functionality for a specific format (e.g. ``DataFrame``).
        batch_size
            Batch size for generator. Yields augmented data points
            for the next ``batch_size`` input data points.

        Yields
        ------
        Any
            Collections of data points in augmented data set for batches of inputs

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError


class TFApplier(BaseTFApplier):
    """TF applier for a list of data points.

    Augments a list of data points (e.g. ``SimpleNamespace``). Primarily
    useful for testing.
    """

    def apply_generator(  # type: ignore
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

    def apply(self, data_points: DataPoints) -> List[DataPoint]:  # type: ignore
        """Augment a list of data points using TFs and policy.

        Parameters
        ----------
        data_points
            List containing data points to be transformed

        Returns
        -------
        List[DataPoint]
            List of data points in augmented data set
        """
        x_transformed: List[DataPoint] = []
        for x in tqdm(data_points):
            x_transformed.extend(self._apply_policy_to_data_point(x))
        return x_transformed
