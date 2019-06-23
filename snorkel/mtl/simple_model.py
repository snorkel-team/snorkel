"""A proof-of-concept that we can create a simple single-task model interface while
primarily using the MultitaskModel class.
"""

from functools import partial
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn

from snorkel.mtl.model import MultitaskModel
from snorkel.mtl.modules.utils import ce_loss, softmax
from snorkel.mtl.scorer import Scorer
from snorkel.mtl.task import Task

# def ce_loss(X, Y):
#     return F.cross_entropy(X, Y - 1)


# def softmax(X):
#     return F.softmax(X, dim=1).cpu().numpy()


class SimpleModel(MultitaskModel):
    def __init__(
        self,
        modules: List[nn.Module],
        dropout: float = 0.0,
        loss_func: Callable[[torch.Tensor], torch.FloatTensor] = ce_loss,
        output_func: Callable[[torch.Tensor], np.ndarray] = softmax,
        metrics: List[str] = ["accuracy"],
        **kwargs,
    ):

        module_pool = {}
        task_flow = []
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
                inputs = [(task_flow[-1]["name"], 0)]
            task_flow.append(
                {"name": f"op{i}", "module": f"module{i}", "inputs": inputs}
            )

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
