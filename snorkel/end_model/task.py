import logging
from typing import Callable, List

import torch
from torch import nn

from snorkel.end_model.scorer import Scorer


class Operation:
    """A single operation to execute in a task flow

    The `name` attributes defaults to `module_name` since most of the time, each module
    is used only once per task flow. For more advanced flows where the same module is
    used multiple times per forward pass, a name may be explicitly given to
    differentiate the Operations.
    """

    def __init__(self, module_name, inputs, name=None):
        self.name = name or module_name
        self.module_name = module_name
        self.inputs = inputs

    def __repr__(self):
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
        loss_func: Callable[..., torch.Tensor],
        output_func: Callable[..., torch.Tensor],
        scorer: Scorer,
    ) -> None:
        self.name = name
        assert isinstance(module_pool, nn.ModuleDict) is True
        self.module_pool = module_pool
        self.task_flow = task_flow
        self.loss_func = loss_func
        self.output_func = output_func
        self.scorer = scorer

        logging.info(f"Created task: {self.name}")

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"
