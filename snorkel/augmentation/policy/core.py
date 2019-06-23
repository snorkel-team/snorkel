from typing import List

import numpy as np


class AugmentationPolicy:
    def __init__(self, n_tfs: int):
        self._n = n_tfs

    def generate(self) -> List[int]:
        raise NotImplementedError


class RandomAugmentationPolicy(AugmentationPolicy):
    def __init__(self, n_tfs: int, sequence_length: int = 1) -> None:
        self._k = sequence_length
        super().__init__(n_tfs)

    def generate(self) -> List[int]:
        return np.random.choice(self._n, size=self._k).tolist()
