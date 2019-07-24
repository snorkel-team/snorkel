from typing import Tuple

from snorkel.types import Config


class SGDOptimizerConfig(Config):
    """Settings for SGD optimizer."""

    momentum: float = 0.9


class AdamOptimizerConfig(Config):
    """Settings for Adam optimizer."""

    amsgrad: bool = False
    betas: Tuple[float, float] = (0.9, 0.999)


class AdamaxOptimizerConfig(Config):
    """Settings for Adamax optimizer."""

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


class OptimizerConfig(Config):
    """Settings common to all optimizers."""

    sgd_config: SGDOptimizerConfig = SGDOptimizerConfig()  # type:ignore
    adam_config: AdamOptimizerConfig = AdamOptimizerConfig()  # type:ignore
    adamax_config: AdamaxOptimizerConfig = AdamaxOptimizerConfig()  # type:ignore
