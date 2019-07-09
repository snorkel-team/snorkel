import logging
from functools import partial
from typing import Callable, List, Sequence, Tuple, Union, Optional

import torch
from torch import nn
import torch.nn.functional as F

from snorkel.classification.scorer import Scorer


class Operation:
    """A single operation to execute in a task flow

    The `name` attributes defaults to `module_name` since most of the time, each module
    is used only once per task flow. For more advanced flows where the same module is
    used multiple times per forward pass, a name may be explicitly given to
    differentiate the Operations.
    """

    def __init__(
        self,
        module_name: str,
        inputs: Sequence[Tuple[str, Union[str, int]]],
        name: str = None,
    ) -> None:
        self.name = name or module_name
        self.module_name = module_name
        self.inputs = inputs

    def __repr__(self) -> str:
        return (
            f"Operation(name={self.name}, "
            f"module_name={self.module_name}, "
            f"inputs={self.inputs}"
        )


class Task:
    """A single Task in a multi-task problem

    :param name: The name of the task (Primary key)
    :param module_pool: A dict of all modules that are used by the task
    :param task_flow: The task flow among modules to define how the data flows
    :param loss_func: The function to calculate the loss
    :param output_func: The function to generate the output
    :param scorer: A Scorer defining the metrics to evaluate on the task
    """

    def __init__(
        self,
        name: str,
        module_pool: nn.ModuleDict,
        task_flow: List[Operation],
        scorer: Scorer,
        loss_func: Optional[Callable[..., torch.Tensor]] = None,
        output_func: Optional[Callable[..., torch.Tensor]] = None,
    ) -> None:
        self.name = name
        assert isinstance(module_pool, nn.ModuleDict) is True
        self.module_pool = module_pool
        self.task_flow = task_flow
        self.loss_func = loss_func or partial(ce_loss, task_flow[-1].name)
        self.output_func = output_func or partial(softmax, task_flow[-1].name)
        self.scorer = scorer

        logging.info(f"Created task: {self.name}")

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"


def ce_loss(module_name, outputs, Y, active):
    # Subtract 1 from hard labels in Y to account for Snorkel reserving the label 0 for
    # abstains while F.cross_entropy() expects 0-indexed labels
    return F.cross_entropy(outputs[module_name][0][active], (Y.view(-1) - 1)[active])


def softmax(module_name, outputs):
    return F.softmax(outputs[module_name][0], dim=1)
