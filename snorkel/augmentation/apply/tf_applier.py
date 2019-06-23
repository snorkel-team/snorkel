from typing import Any, List

from tqdm import tqdm

from snorkel.augmentation.policy import AugmentationPolicy
from snorkel.augmentation.tf import (
    BaseTransformationFunction,
    TransformationFunctionMode,
)
from snorkel.types import DataPoint


class BaseTFApplier:
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
        raise NotImplementedError


class TFApplier(BaseTFApplier):
    def apply(self, data_points: List[DataPoint]) -> List[DataPoint]:  # type: ignore
        self._set_tf_mode(TransformationFunctionMode.NAMESPACE)
        x_transformed = []
        for x in tqdm(data_points):
            x_transformed.extend(self._apply_policy_to_data_point(x))
        return x_transformed
