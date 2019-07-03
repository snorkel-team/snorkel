import logging
import os
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn

from snorkel.analysis.utils import probs_to_preds

from .scorer import Scorer
from .snorkel_config import default_config
from .task import Operation, Task
from .utils import move_to_device, recursive_merge_dicts


class MultitaskModel(nn.Module):
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

    def _move_to_device(self):
        """Move model to specified device."""

        if self.config["device"] >= 0:
            if torch.cuda.is_available():
                logging.info(f"Moving model to GPU " f"(cuda:{self.config['device']}).")
                self.to(torch.device(f"cuda:{self.config['device']}"))
            else:
                logging.info("No cuda device available. Switch to cpu instead.")

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
        self, X_dict: Mapping[Union[str, int], Any], task_names: Iterable[str]
    ) -> Dict[str, Mapping[Union[str, int], Any]]:
        """Forward pass through the network

        :param X_dict: The input data
        :param task_names: The task names that needs to forward
        :return: The output of all forwarded modules
        """

        X_dict = move_to_device(X_dict, self.config["device"])

        outputs: Dict[str, Mapping[Union[str, int], Any]] = {}
        outputs["_input_"] = X_dict

        # Call forward for each task, using cached result if available
        # Each task flow consists of one or more operations that are executed in order
        for task_name in task_names:
            task_flow = self.task_flows[task_name]

            for operation in task_flow:
                if operation.name not in outputs:
                    if operation.inputs:
                        # Feed the inputs the module requested in the reqested order
                        try:
                            input = [
                                outputs[operation_name][output_index]
                                for operation_name, output_index in operation.inputs
                            ]
                        except Exception:
                            raise ValueError(f"Unrecognized operation {operation}.")
                        output = self.module_pool[operation.module_name].forward(*input)
                    else:
                        # Feed the entire outputs dict for the module to pull
                        output = self.module_pool[operation.module_name].forward(
                            outputs
                        )
                    if isinstance(output, tuple):
                        output = list(output)
                    if not isinstance(output, list):
                        output = [output]
                    outputs[operation.name] = output

        return outputs

    def calculate_loss(
        self,
        X_dict: Mapping[Union[str, int], torch.Tensor],
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
    def _calculate_probs(
        self, X_dict: Mapping[Union[str, int], torch.Tensor], task_names: Iterable[str]
    ):
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

    def save(self, model_path):
        """Save the current model
        :param model_path: Saved model path.
        :type model_path: str
        """

        # Check existence of model saving directory and create if does not exist.
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        state_dict = {
            "model": {"name": self.name, "module_pool": self.collect_state_dict()}
        }

        try:
            torch.save(state_dict, model_path)
        except BaseException:
            logging.warning("Saving failed... continuing anyway.")

        logging.info(f"[{self.name}] Model saved in {model_path}")

    def load(self, model_path):
        """Load model state_dict from file and reinitialize the model weights.
        :param model_path: Saved model path.
        :type model_path: str
        """

        if not os.path.exists(model_path):
            logging.error("Loading failed... Model does not exist.")

        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        except BaseException:
            logging.error(f"Loading failed... Cannot load model from {model_path}")
            raise

        self.load_state_dict(checkpoint["model"]["module_pool"])

        logging.info(f"[{self.name}] Model loaded from {model_path}")

        # Move model to specified device
        self._move_to_device()

    def collect_state_dict(self):
        state_dict = defaultdict(list)

        for module_name, module in self.module_pool.items():
            if self.config["dataparallel"]:
                state_dict[module_name] = module.module.state_dict()
            else:
                state_dict[module_name] = module.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):

        for module_name, module_state_dict in state_dict.items():
            if module_name in self.module_pool:
                if self.config["dataparallel"]:
                    self.module_pool[module_name].module.load_state_dict(
                        module_state_dict
                    )
                else:
                    self.module_pool[module_name].load_state_dict(module_state_dict)
            else:
                logging.info(f"Missing {module_name} in module_pool, skip it..")
