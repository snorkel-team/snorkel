from typing import Any, List

from tqdm import tqdm

from snorkel.augmentation.policy import AugmentationPolicy
from snorkel.augmentation.tf import (
    BaseTransformationFunction,
    TransformationFunctionMode,
)
from snorkel.types import DataPoint


class BaseTFApplier:
    """Base class for TF applier objects.

    Base class for TF applier objects, which execute a set of TF
    on a collection of data points. Subclasses should operate on
    a single data point collection format (e.g. `DataFrame`).
    Subclasses must implement the `apply` method.

    Parameters
    ----------
    tfs
        TFs that this applier executes on examples
    policy
        Augmentation policy used to generate sequences of TFs
    k
        Number of transformed data points per original
    keep_original
        Keep untransformed data point in augmented data set?

    Raises
    ------
    NotImplementedError
        `apply` method must be implemented by subclasses
    """

    def __init__(
        self,
        tfs: List[BaseTransformationFunction],
        policy: AugmentationPolicy,
        k: int = 1,
        keep_original: bool = True,
    ) -> None:
        self._tfs = tfs
        self._policy = policy
        self._k = k
        self._keep_original = keep_original

    def _set_tf_mode(self, mode: TransformationFunctionMode) -> None:
        for tf in self._tfs:
            tf.set_mode(mode)

    def _apply_policy_to_data_point(self, x: DataPoint) -> List[DataPoint]:
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
        """Label collection of data points with LFs.

        Parameters
        ----------
        data_points
            Collection of data points to be transformed by TFs and policy. Subclasses
            implement functionality for a specific format (e.g. `DataFrame`).

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


class TFApplier(BaseTFApplier):
    """TF applier for a list of data points.

    Augments a list of data points (e.g. `SimpleNamespace`). Primarily
    useful for testing.
    """

    def apply(self, data_points: List[DataPoint]) -> List[DataPoint]:  # type: ignore
        """Augment a list of data points using TFs and policy.

        Parameters
        ----------
        data_points
            List containing data points to be transformed

        Returns
        -------
        List[DataPoint]
            Augmented list of data points
        """
        self._set_tf_mode(TransformationFunctionMode.NAMESPACE)
        x_transformed = []
        for x in tqdm(data_points):
            x_transformed.extend(self._apply_policy_to_data_point(x))
        return x_transformed
