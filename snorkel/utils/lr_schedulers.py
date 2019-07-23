from snorkel.types import Config


class ExponentialLRSchedulerConfig(Config):
    """Settings for Exponential decay learning rate scheduler."""

    gamma: float = 0.9


class StepLRSchedulerConfig(Config):
    """Settings for Step decay learning rate scheduler."""

    gamma: float = 0.9
    step_size: int = 5


class LRSchedulerConfig(Config):
    """Settings common to all LRSchedulers.

    Parameters
    ----------
    warmup_steps
        The number of warmup_units over which to perform learning rate warmup (a linear
        increase from 0 to the specified lr)
    warmup_unit
        The unit to use when counting warmup (one of ["batches", "epochs"])
    warmup_percentage
        The percentage of the training procedure to warm up over (ignored if
        warmup_steps is non-zero)
    min_lr
        The minimum learning rate to use during training (the learning rate specified
        by a learning rate scheduler will be rounded up to this if it is lower)
    exponential_config
        Extra settings for the ExponentialLRScheduler
    step_config
        Extra settings for the StepLRScheduler
    """

    warmup_steps: float = 0  # warm up steps
    warmup_unit: str = "batches"  # [epochs, batches]
    warmup_percentage: float = 0.0  # warm up percentage
    min_lr: float = 0.0  # minimum learning rate
    exponential_config: ExponentialLRSchedulerConfig = ExponentialLRSchedulerConfig()  # type:ignore
    step_config: StepLRSchedulerConfig = StepLRSchedulerConfig()  # type:ignore
