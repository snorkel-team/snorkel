"""PyTorch-based multi-task learning framework for discriminative modeling."""

from .data import DictDataLoader, DictDataset  # noqa: F401
from .loss import (  # noqa: F401
    cross_entropy_from_outputs,
    cross_entropy_with_probs,
    cross_entropy_with_probs_from_outputs,
)
from .multitask_classifier import MultitaskClassifier  # noqa: F401
from .task import Operation, Task, softmax_from_outputs  # noqa: F401
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
