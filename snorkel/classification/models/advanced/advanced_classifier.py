from functools import partial
import logging
import os
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Union, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from snorkel.analysis.utils import probs_to_preds
from snorkel.classification.scorer import Scorer
from snorkel.classification.snorkel_config import default_config
from snorkel.classification.utils import move_to_device, recursive_merge_dicts

from .task import Operation, Task
from .utils import ce_loss, softmax


class AdvancedClassifier(nn.Module):
    """A class to build multi-task model.

    :param name: Name of the model
    :param tasks: a list of Tasks to be trained jointly
    """

    def __init__(
        self, tasks: List[Task], name: Optional[str] = None, **kwargs: Any
    ) -> None:
        super().__init__()
        self.config = recursive_merge_dicts(
            default_config["model_config"], kwargs, misses="insert"
        )
        self.name = name or type(self).__name__

        # Initiate the model attributes
        self.module_pool = nn.ModuleDict()
        self.task_names: Set[str] = set()
        self.task_flows: Dict[str, List[Operation]] = dict()
        self.loss_funcs: Dict[str, Callable[..., torch.Tensor]] = dict()
        self.output_funcs: Dict[str, Callable[..., torch.Tensor]] = dict()
        self.scorers: Dict[str, Scorer] = dict()

        # Build network with given tasks
        self._build_network(tasks)

        logging.info(
            f"Created multi-task model {self.name} that contains "
            f"task(s) {self.task_names}."
        )

        # Move model to specified device
        self._move_to_device()

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"

    def _build_network(self, tasks: List[Task]) -> None:
        """Build the MTL network using all tasks"""

        for task in tasks:
            if task.name in self.task_names:
                raise ValueError(
                    f"Found duplicate task {task.name}, different task should use "
                    f"different task name."
                )
            if not isinstance(task, Task):
                raise ValueError(f"Unrecognized task type {task}.")
            self.add_task(task)

    def add_task(self, task):
        """Add a single task into MTL network"""

        # Combine module_pool from all tasks
        for key in task.module_pool.keys():
            if key in self.module_pool.keys():
                if self.config["dataparallel"]:
                    task.module_pool[key] = nn.DataParallel(self.module_pool[key])
                else:
                    task.module_pool[key] = self.module_pool[key]
            else:
                if self.config["dataparallel"]:
                    self.module_pool[key] = nn.DataParallel(task.module_pool[key])
                else:
                    self.module_pool[key] = task.module_pool[key]
        # Collect task names
        self.task_names.add(task.name)
        # Collect task flows
        self.task_flows[task.name] = task.task_flow
        # Collect loss functions
        self.loss_funcs[task.name] = task.loss_func
        # Collect output functions
        self.output_funcs[task.name] = task.output_func
        # Collect scorers
        self.scorers[task.name] = task.scorer

        # Move model to specified device
        self._move_to_device()

    def forward(  # type: ignore
        self, X_dict: Mapping[str, Any], task_names: Iterable[str]
    ) -> Any:
        """Forward pass through the network

        :param X_dict: The input data
        :param task_names: The task names that needs to forward
        :return: The output of all forwarded modules
        """

        X_dict = move_to_device(X_dict, self.config["device"])

        outputs: Dict[str, Mapping[Union[str, int], Any]] = {}
        outputs["_input_"] = X_dict  # type: ignore

        # Call forward for each task, using cached result if available
        # Each task flow consists of one or more operations that are executed in order
        for task_name in task_names:
            task_flow = self.task_flows[task_name]

            for operation in task_flow:
                if operation.name not in outputs:
                    if operation.inputs:
                        # Feed the inputs the module requested in the reqested order
                        try:
                            inputs = []
                            for op_input in operation.inputs:
                                if isinstance(op_input[1], int):
                                    # The output of the indicated operation has only
                                    # one field; use that as the input to the current op
                                    op_name, field_idx = op_input
                                    inputs.append(outputs[op_name][field_idx])
                                elif isinstance(op_input[1], str):
                                    # The output of the indicated operation has a dict
                                    # of fields; extract the designated field by name
                                    op_name, field_key = op_input
                                    inputs.append(outputs[op_name][field_key])
                                else:
                                    raise ValueError(
                                        f"Invalid input to operation {operation}: "
                                        f"{op_input}. Expected an input specification "
                                        f"of type (str, int) or (str, str)."
                                    )
                        except Exception:
                            raise ValueError(f"Unrecognized operation {operation}.")
                        output = self.module_pool[operation.module_name].forward(
                            *inputs
                        )
                    else:
                        # Feed the entire outputs dict for the module to pull from
                        # TODO: Remove this option (only slice combiner module uses it)
                        output = self.module_pool[operation.module_name].forward(
                            outputs
                        )
                    if not isinstance(output, Mapping):
                        # Make output a singleton list so it becomes a valid Mapping
                        output = [output]
                    outputs[operation.name] = output

        # Note: We return all outputs to enable advanced workflows such as multi-task
        # learning (where we want to calculate loss from multiple head modules on
        # forward passes).
        return outputs

    def calculate_loss(
        self,
        X_dict: Mapping[str, Any],
        Y_dict: Dict[str, torch.Tensor],
        task_to_label_dict: Dict[str, str],
    ):
        """Calculate the loss

        :param X_dict: The input data
        :param Y_dict: The output data
        :param task_to_label_dict: The task to label mapping
        :return: The loss and the number of samples in the batch of all tasks
        :rtype: dict, dict
        """

        loss_dict = dict()
        count_dict = dict()

        task_names = task_to_label_dict.keys()
        outputs = self.forward(X_dict, task_names)

        # Calculate loss for each task
        for task_name, label_name in task_to_label_dict.items():

            Y = Y_dict[label_name]

            # Select the active samples
            if len(Y.size()) == 1:
                active = Y.detach() != 0
            else:
                active = torch.any(Y.detach() != 0, dim=1)

            # Only calculate the loss when active example exists
            if active.any():
                count_dict[task_name] = active.sum().item()

                loss_dict[task_name] = self.loss_funcs[task_name](
                    outputs,
                    move_to_device(Y, self.config["device"]),
                    move_to_device(active, self.config["device"]),
                )

        return loss_dict, count_dict

    @torch.no_grad()
    def _calculate_probs(self, X_dict: Mapping[str, Any], task_names: Iterable[str]):
        """Calculate the probs given the features

        :param X_dict: The input data
        :param task_names: The task names that needs to forward
        """

        self.eval()

        prob_dict = dict()

        outputs = self.forward(X_dict, task_names)

        # Calculate prediction for each task
        for task_name in task_names:
            prob_dict[task_name] = self.output_funcs[task_name](outputs).cpu().numpy()

        return prob_dict

    @torch.no_grad()
    def predict(self, dataloader, return_preds=False):

        self.eval()

        gold_dict = defaultdict(list)
        prob_dict = defaultdict(list)

        for batch_num, (X_batch_dict, Y_batch_dict) in enumerate(dataloader):
            prob_batch_dict = self._calculate_probs(
                X_batch_dict, dataloader.task_to_label_dict.keys()
            )
            for task_name in dataloader.task_to_label_dict.keys():
                prob_dict[task_name].extend(prob_batch_dict[task_name])
                gold_dict[task_name].extend(
                    Y_batch_dict[dataloader.task_to_label_dict[task_name]].cpu().numpy()
                )
        for task_name in gold_dict:
            gold_dict[task_name] = np.array(gold_dict[task_name])
            prob_dict[task_name] = np.array(prob_dict[task_name])
            if len(gold_dict[task_name].shape) == 1:
                active = (gold_dict[task_name] != 0).reshape(-1)
            else:
                active = np.sum(gold_dict[task_name] == 0, axis=1) > 0

            if 0 in active:
                gold_dict[task_name] = gold_dict[task_name][active]
                prob_dict[task_name] = prob_dict[task_name][active]

        if return_preds:
            pred_dict = defaultdict(list)
            for task_name, probs in prob_dict.items():
                pred_dict[task_name] = probs_to_preds(probs)

        results = {"golds": gold_dict, "probs": prob_dict}

        if return_preds:
            results["preds"] = pred_dict

        return results

    @torch.no_grad()
    def score(self, dataloaders):
        """Score the data from dataloader with the model

        :param dataloaders: the dataloader that performs scoring
        :type dataloaders: dataloader
        """

        self.eval()

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        metric_score_dict = dict()

        for dataloader in dataloaders:
            results = self.predict(dataloader, return_preds=True)
            for task_name in results["golds"].keys():
                metric_scores = self.scorers[task_name].score(
                    results["golds"][task_name],
                    results["preds"][task_name],
                    results["probs"][task_name],
                )
                for metric_name, metric_value in metric_scores.items():
                    identifier = "/".join(
                        [
                            task_name,
                            dataloader.dataset.name,
                            dataloader.dataset.split,
                            metric_name,
                        ]
                    )
                    metric_score_dict[identifier] = metric_value

        return metric_score_dict

    def save(self, model_path: str):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        try:
            torch.save(self.state_dict(), model_path)
        except BaseException:
            logging.warning("Saving failed... continuing anyway.")

        logging.info(f"[{self.name}] Model saved in {model_path}")

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            logging.error("Loading failed... Model does not exist.")

        try:
            self.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
        except BaseException:
            logging.error(f"Loading failed... Cannot load model from {model_path}")
            raise

        logging.info(f"[{self.name}] Model loaded from {model_path}")
        self._move_to_device()

    def _move_to_device(self):
        device = self.config["device"]
        if device >= 0:
            if torch.cuda.is_available():
                logging.info(f"Moving model to GPU (cuda:{device}).")
                self.to(torch.device(f"cuda:{device}"))
            else:
                logging.info("No cuda device available. Switch to cpu instead.")

    @classmethod
    def from_modules(
        self,
        modules: List[nn.Module],
        dropout: float = 0.0,
        loss_func: Callable[..., torch.FloatTensor] = ce_loss,
        output_func: Callable[..., np.ndarray] = softmax,
        metrics: List[str] = ["accuracy"],
        **kwargs,
    ):
        """Define a simple AdvancedClassifier from `Module`s instead of `Tasks

        TBD
        """
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
        return AdvancedClassifier([task], **kwargs)