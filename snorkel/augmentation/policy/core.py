import random
from typing import List

from snorkel.augmentation.tf import TransformationFunction, TransformationFunctionMode
from snorkel.types import DataPoint


class AugmentationPolicy:
    def __init__(self, tfs: List[TransformationFunction]):
        self._tfs = tfs

    def set_tf_mode(self, mode: TransformationFunctionMode) -> None:
        for tf in self._tfs:
            tf.set_mode(mode)

    def apply(self, x: DataPoint) -> DataPoint:
        raise NotImplementedError


class RandomAugmentationPolicy(AugmentationPolicy):
    def __init__(
        self, tfs: List[TransformationFunction], sequence_length: int = 1
    ) -> None:
        self._k = sequence_length
        super().__init__(tfs)

    def apply(self, x: DataPoint) -> DataPoint:
        for _ in range(self._k):
            tf = random.choice(self._tfs)
            x = tf(x)
        return x
