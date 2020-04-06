from typing import Dict, List

import numpy as np
import torch
from numpy.lib import recfunctions as rfn
from torch import nn

from snorkel.analysis import Scorer
from snorkel.classification.data import DictDataLoader
from snorkel.classification.multitask_classifier import Operation, Task

from .modules.slice_combiner import SliceCombinerModule


def add_slice_labels(
    dataloader: DictDataLoader, base_task: Task, S: np.recarray
) -> None:
    """Modify a dataloader in-place, adding labels for slice tasks.

    Parameters
    ----------
    dataloader
        A DictDataLoader whose dataset.Y_dict attribute will be modified in place
    base_task
       The Task for which we want corresponding slice tasks/labels
    S
        A recarray (output of SFApplier) containing data fields with slice
        indicator information
    """
    # Add the base task if it's missing
    if "base" not in S.dtype.names:
        # Create a new np.recarray with an additional "base" data field
        S = rfn.append_fields(
            [S], names=[("base")], data=[np.ones(S.shape)], asrecarray=True
        )

    slice_names = S.dtype.names

    Y_dict: Dict[str, np.ndarray] = dataloader.dataset.Y_dict  # type: ignore
    labels = Y_dict[base_task.name]

    for slice_name in slice_names:
        # Gather ind labels
        ind_labels = torch.LongTensor(S[slice_name])  # type: ignore

        # Mask out "inactive" pred_labels as specified by ind_labels
        pred_labels = labels.clone()
        pred_labels[~ind_labels.bool()] = -1

        ind_task_name = f"{base_task.name}_slice:{slice_name}_ind"
        pred_task_name = f"{base_task.name}_slice:{slice_name}_pred"

        # Update dataloaders
        Y_dict[ind_task_name] = ind_labels
        Y_dict[pred_task_name] = pred_labels


def convert_to_slice_tasks(base_task: Task, slice_names: List[str]) -> List[Task]:
    """Add slice labels to dataloader and creates new slice tasks (including base slice).

    Each slice will get two slice-specific heads:
    - an indicator head that learns to identify when DataPoints are in that slice
    - a predictor head that is trained on only members of that slice

    The base task's head is replaced by a master head that makes predictions based on
    a combination of the predictor heads' predictions that are weighted by the
    indicator heads' prediction confidences.

    WARNING: The current implementation pollutes the module_pool---the indicator task's
    module_pool includes predictor modules and vice versa since both are modified in
    place. This does not affect the result because the op sequences dictate which modules
    get used, and those do not include the extra modules. An alternative would be to
    make separate copies of the module pool for each, but that wastes time and memory
    extra copies of (potentially very large) modules that will be merged in a moment
    away in the model since they have the same name. We leave resolution of this issue
    for a future release.


    Parameters
    ----------
    base_task
        Task for which we are adding slice tasks. As noted in the WARNING, this task's
        module_pool will currently be modified in place for efficiency purposes.
    slice_names
        List of slice names corresponding to the columns of the slice matrix.

    Returns
    -------
    List[Task]
        Containins original base_task, pred/ind tasks for the base slice, and pred/ind
        tasks for each of the specified slice_names
    """

    if "base" not in slice_names:
        slice_names = slice_names + ["base"]

    slice_tasks = []

    # Keep track of all operations related to slice tasks
    slice_task_ops = []

    # NOTE: We assume here that the last operation uses the head module
    # Identify base task head module
    head_module_op = base_task.op_sequence[-1]
    head_module = base_task.module_pool[head_module_op.module_name]

    if isinstance(head_module, nn.DataParallel):
        head_module = head_module.module

    neck_size = head_module.in_features
    base_task_cardinality = head_module.out_features

    # Remove the slice-unaware head module from module pool and op sequence
    del base_task.module_pool[head_module_op.module_name]
    body_flow = base_task.op_sequence[:-1]

    # Create slice indicator tasks
    for slice_name in slice_names:

        ind_task_name = f"{base_task.name}_slice:{slice_name}_ind"
        ind_head_module_name = f"{ind_task_name}_head"
        # Indicator head always predicts "in the slice or not", so is always binary
        ind_head_module = nn.Linear(neck_size, 2)

        # Create module_pool
        ind_module_pool = base_task.module_pool
        ind_module_pool[ind_head_module_name] = ind_head_module

        # Define operations for task head
        ind_head_op = Operation(
            module_name=ind_head_module_name, inputs=head_module_op.inputs
        )
        ind_task_ops = [ind_head_op]
        slice_task_ops.extend(ind_task_ops)

        # Create op sequence
        ind_op_sequence = list(body_flow) + list(ind_task_ops)

        # Create ind task
        ind_task = Task(
            name=ind_task_name,
            module_pool=ind_module_pool,
            op_sequence=ind_op_sequence,
            # NOTE: F1 by default because indicator task is often class imbalanced
            scorer=Scorer(metrics=["f1"]),
        )
        slice_tasks.append(ind_task)

    # Create slice predictor tasks
    shared_pred_head_module = nn.Linear(neck_size, base_task_cardinality)
    for slice_name in slice_names:

        pred_task_name = f"{base_task.name}_slice:{slice_name}_pred"

        pred_head_module_name = f"{pred_task_name}_head"
        pred_transform_module_name = f"{pred_task_name}_transform"
        pred_transform_module = nn.Linear(neck_size, neck_size)

        # Create module_pool
        # NOTE: See note in doc string about module_pool polution
        pred_module_pool = base_task.module_pool
        pred_module_pool[pred_transform_module_name] = pred_transform_module
        pred_module_pool[pred_head_module_name] = shared_pred_head_module

        # Define operations for task head
        pred_transform_op = Operation(
            module_name=pred_transform_module_name, inputs=head_module_op.inputs
        )
        pred_head_op = Operation(
            module_name=pred_head_module_name, inputs=[pred_transform_op.name]
        )
        pred_task_ops = [pred_transform_op, pred_head_op]
        slice_task_ops.extend(pred_task_ops)

        # Create op sequence
        pred_op_sequence = list(body_flow) + list(pred_task_ops)

        # Create pred task
        pred_task = Task(
            name=pred_task_name,
            module_pool=pred_module_pool,
            op_sequence=pred_op_sequence,
            scorer=base_task.scorer,
        )
        slice_tasks.append(pred_task)

    # Create master task
    master_task_name = base_task.name
    master_combiner_module_name = f"{base_task.name}_master_combiner"
    master_combiner_module = SliceCombinerModule()
    master_head_module_name = f"{base_task.name}_master_head"
    master_head_module = head_module

    # Create module_pool
    master_module_pool = nn.ModuleDict(
        {
            master_combiner_module_name: master_combiner_module,
            master_head_module_name: master_head_module,
        }
    )

    master_combiner_op = Operation(module_name=master_combiner_module_name, inputs=[])
    master_head_op = Operation(
        module_name=master_head_module_name, inputs=[master_combiner_op.name]
    )

    # NOTE: See note in doc string about module_pool polution

    # Create op_sequence
    master_op_sequence = (
        list(body_flow) + list(slice_task_ops) + [master_combiner_op, master_head_op]
    )

    master_task = Task(
        name=master_task_name,
        module_pool=master_module_pool,
        op_sequence=master_op_sequence,
        scorer=base_task.scorer,
    )
    return slice_tasks + [master_task]
