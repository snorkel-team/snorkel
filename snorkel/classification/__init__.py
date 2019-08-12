"""PyTorch-based multi-task learning framework for discriminative modeling."""

from .data import DictDataLoader, DictDataset  # noqa: F401
from .loss import cross_entropy_with_probs  # noqa: F401
from .multitask_classifier import MultitaskClassifier  # noqa: F401
from .task import Operation, Task  # noqa: F401
from .training.loggers import (  # noqa: F401
    Checkpointer,
    CheckpointerConfig,
    LogManager,
    LogManagerConfig,
    LogWriter,
    LogWriterConfig,
    TensorBoardWriter,
)
from .training.trainer import Trainer  # noqa: F401
