"""PyTorch-based multi-task learning framework for discriminative modeling."""

from .data import DictDataLoader, DictDataset  # noqa: F401
from .loss import (  # noqa: F401
    ce_loss_from_outputs,
    cross_entropy_soft_from_outputs,
    cross_entropy_soft,
)
from .scorer import Scorer  # noqa: F401
from .snorkel_classifier import SnorkelClassifier  # noqa: F401
from .task import (  # noqa: F401
    Operation,
    Task,
    softmax_from_outputs,
)
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
