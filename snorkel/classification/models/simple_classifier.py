"""A proof-of-concept that we can create a simple single-task model interface while
primarily using the AdvancedClassifier class.
"""

from functools import partial
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from snorkel.classification.models.advanced import AdvancedClassifier, Operation, Task
from snorkel.classification.models.advanced.utils import ce_loss, softmax
from snorkel.classification.scorer import Scorer


class SimpleClassifier(AdvancedClassifier):
    def __init__(
        self,
        modules: List[nn.Module],
        dropout: float = 0.0,
        loss_func: Callable[..., torch.FloatTensor] = ce_loss,
        output_func: Callable[..., np.ndarray] = softmax,
        metrics: List[str] = ["accuracy"],
        **kwargs,
    ):

        module_pool: Dict[str, nn.Module] = {}
        task_flow: List[Operation] = []
        inputs: Sequence[Tuple[str, Union[str, int]]]
        for i, module in enumerate(modules):

            args = [module]
            if i < len(modules) - 1:
                args.append(nn.ReLU())
                if dropout > 0:
                    args.append(nn.Dropout(dropout))
            module_block = nn.Sequential(*args)

            module_pool[f"module{i}"] = module_block
            if i == 0:
                inputs = [("_input_", "data")]
            else:
                inputs = [(task_flow[-1].name, 0)]
            op = Operation(name=f"op{i}", module_name=f"module{i}", inputs=inputs)
            task_flow.append(op)

        last_op = f"op{len(modules) - 1}"
        task = Task(
            name="task",
            module_pool=nn.ModuleDict(module_pool),
            task_flow=task_flow,
            loss_func=partial(ce_loss, last_op),
            output_func=partial(output_func, last_op),
            scorer=Scorer(metrics=metrics),
        )
        super().__init__([task], **kwargs)
