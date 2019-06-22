import logging
from functools import partial
from typing import List

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import nn

from snorkel.mtl.data import MultitaskDataLoader
from snorkel.mtl.modules.utils import ce_loss, softmax
from snorkel.mtl.scorer import Scorer
from snorkel.mtl.task import Task

from .modules.slice_combiner import SliceCombinerModule


def _add_base_slice(slice_labels, slice_names):
    # Add base slice
    if "base" not in slice_names:
        num_points, num_slices = slice_labels.shape
        base_labels = np.ones((num_points, 1), dtype=int)
        slice_labels = np.hstack([slice_labels, base_labels])
        # Make a copy so we don't modify in place
        slice_names = slice_names + ["base"]
    return slice_labels, slice_names


def add_slice_labels(
    base_task: Task,
    dataloader: MultitaskDataLoader,
    slice_labels: csr_matrix,  # [n, m]
    slice_names: List[str],
):
    slice_labels = slice_labels.toarray()
    slice_labels, slice_names = _add_base_slice(slice_labels, slice_names)

    label_name = dataloader.task_to_label_dict[base_task.name]
    labels = dataloader.dataset.Y_dict[label_name]
    for i, slice_name in enumerate(slice_names):

        # Convert labels
        ind_labels = torch.LongTensor(slice_labels[:, i])  # [n, 1]
        pred_labels = ind_labels * labels

        ind_task_name = f"{base_task.name}_slice:{slice_name}_ind"
        pred_task_name = f"{base_task.name}_slice:{slice_name}_pred"

        # Update dataloaders
        dataloader.dataset.Y_dict[ind_task_name] = ind_labels
        dataloader.dataset.Y_dict[pred_task_name] = pred_labels

        dataloader.task_to_label_dict[ind_task_name] = ind_task_name
        dataloader.task_to_label_dict[pred_task_name] = pred_task_name
    return dataloader


def convert_to_slice_tasks(base_task: Task, slice_names: List[str]):
    """Adds slice labels to dataloader and creates new slice tasks (including base slice)
    
    Each slice will get two slice-specific heads:
    - an indicator head that learns to identify when DataPoints are in that slice
    - a predictor head that is trained on only members of that slice
    
    The base task's head is replaced by a master head that makes predictions based on 
    a combination of the predictor heads' predictions that are weighted by the 
    indicator heads' predictions.
    
    NOTE: The current implementation pollutes the module_pool---the indicator's task's
    module_pool includes predictor modules and vice versa. This does not affect the 
    result because the task flows dictate which modules get used, and those do not 
    include the extra modules. An alternative would be to make separate copies of the 
    module pool for each, but that wastes time and memory extra copies of (potentially 
    very large) modules that will be merged in a moment away in the model since they
    have the same name. We leave resolution of this issue for a future release.
    """
    if "base" not in slice_names:
        slice_names = slice_names + ["base"]

    tasks = []

    # NOTE: We assume here that the last operation uses the head module
    # Identify base task head module
    head_module_op = base_task.task_flow[-1]
    head_module_name = head_module_op["module"]
    head_module = base_task.module_pool[head_module_name]
    head_module_inputs = head_module_op["inputs"]

    if isinstance(head_module, nn.DataParallel):
        head_module = head_module.module

    neck_size = head_module.in_features
    base_task_cardinality = head_module.out_features

    # Remove the slice-unaware head module from module pool and task flow
    del base_task.module_pool[head_module_name]
    base_task.task_flow = base_task.task_flow[:-1]

    # Create slice indicator tasks
    for slice_name in slice_names:

        ind_task_name = f"{base_task.name}_slice:{slice_name}_ind"
        ind_head_module_name = f"{ind_task_name}_ind_head"
        ind_head_module = nn.Linear(neck_size, 2)
        ind_head_op_name = ind_head_module_name

        ind_module_pool = base_task.module_pool
        ind_module_pool[ind_head_module_name] = ind_head_module

        ind_task_flow = base_task.task_flow + [
            {
                "name": ind_head_op_name,
                "module": ind_head_module_name,
                "inputs": head_module_inputs,
            }
        ]

        ind_task = Task(
            name=ind_task_name,
            module_pool=ind_module_pool,
            task_flow=ind_task_flow,
            loss_func=partial(ce_loss, ind_head_op_name),
            output_func=partial(softmax, ind_head_op_name),
            scorer=Scorer(metrics=["accuracy"]),
        )
        tasks.append(ind_task)

    # Create slice predictor tasks
    shared_pred_head_module = nn.Linear(neck_size, base_task_cardinality)
    for slice_name in slice_names:

        pred_task_name = f"{base_task.name}_slice:{slice_name}_pred"
        pred_head_module_name = f"{pred_task_name}_pred_head"
        pred_head_op_name = pred_head_module_name

        pred_transform_module_name = f"{pred_task_name}_pred_transform"
        pred_transform_module = nn.Linear(neck_size, neck_size)
        pred_transform_op_name = pred_transform_module_name

        # NOTE: See note in doc string about module_pool polution
        pred_module_pool = base_task.module_pool
        pred_module_pool[pred_head_module_name] = shared_pred_head_module
        pred_module_pool[pred_transform_module_name] = pred_transform_module

        pred_task_flow = base_task.task_flow + [
            {
                "name": pred_transform_op_name,
                "module": pred_transform_module_name,
                "inputs": head_module_inputs,
            },
            {
                "name": pred_head_op_name,
                "module": pred_head_module_name,
                "inputs": [(pred_transform_op_name, 0)],
            },
        ]

        pred_task = Task(
            name=pred_task_name,
            module_pool=pred_module_pool,
            task_flow=pred_task_flow,
            loss_func=partial(ce_loss, pred_head_op_name),
            output_func=partial(softmax, pred_head_op_name),
            scorer=base_task.scorer,
        )
        tasks.append(pred_task)

    # Create master task
    master_task_name = base_task.name
    master_combiner_module_name = f"{base_task.name}_master_combiner"
    master_combiner_module = SliceCombinerModule()
    master_combiner_op_name = master_combiner_module_name

    master_head_module_name = f"{base_task.name}_master_head"
    master_head_module = nn.Linear(neck_size, base_task_cardinality)
    master_head_op_name = master_head_module_name

    # NOTE: See note in doc string about module_pool polution
    master_module_pool = nn.ModuleDict(
        {
            master_combiner_module_name: master_combiner_module,
            master_head_module_name: master_head_module,
        }
    )

    master_task_flow = (
        ind_task_flow
        + pred_task_flow
        + [
            {
                "name": master_combiner_op_name,
                "module": master_combiner_module_name,
                "inputs": [],  # Module pulls directly from outputs dict
            },
            {
                "name": master_head_op_name,
                "module": master_head_module_name,
                "inputs": [(master_combiner_op_name, 0)],
            },
        ]
    )

    master_task = Task(
        name=master_task_name,
        module_pool=master_module_pool,
        task_flow=master_task_flow,
        loss_func=partial(ce_loss, master_head_op_name),
        output_func=partial(softmax, master_head_op_name),
        scorer=base_task.scorer,
    )
    tasks.append(master_task)

    return tasks
