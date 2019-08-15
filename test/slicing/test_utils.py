import unittest

import pandas as pd
import torch
import torch.nn as nn

from snorkel.classification import DictDataLoader, DictDataset, Operation, Task
from snorkel.slicing import (
    PandasSFApplier,
    add_slice_labels,
    convert_to_slice_tasks,
    slicing_function,
)


@slicing_function()
def f(x):
    return x.val < 0.25


class UtilsTest(unittest.TestCase):
    def test_add_slice_labels(self):
        # Create dummy data
        # Given slicing function f(), we expect the first two entries to be active
        x = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        y = torch.Tensor([0, 1, 1, 0, 1]).long()
        dataset = DictDataset(
            name="TestData", split="train", X_dict={"data": x}, Y_dict={"TestTask": y}
        )

        # Ensure that we start with 1 labelset
        self.assertEqual(len(dataset.Y_dict), 1)

        # Apply SFs with PandasSFApplier
        df = pd.DataFrame({"val": x, "y": y})
        slicing_functions = [f]
        applier = PandasSFApplier(slicing_functions)
        S = applier.apply(df, progress_bar=False)

        dataloader = DictDataLoader(dataset)

        dummy_task = create_dummy_task(task_name="TestTask")
        add_slice_labels(dataloader, dummy_task, S)

        # Ensure that all the fields are present
        labelsets = dataloader.dataset.Y_dict
        self.assertIn("TestTask", labelsets)
        self.assertIn("TestTask_slice:base_ind", labelsets)
        self.assertIn("TestTask_slice:base_pred", labelsets)
        self.assertIn("TestTask_slice:f_ind", labelsets)
        self.assertIn("TestTask_slice:f_pred", labelsets)
        self.assertEqual(len(labelsets), 5)

        # Ensure "ind" contains mask
        self.assertEqual(
            labelsets["TestTask_slice:f_ind"].numpy().tolist(), [1, 1, 0, 0, 0]
        )
        self.assertEqual(
            labelsets["TestTask_slice:base_ind"].numpy().tolist(), [1, 1, 1, 1, 1]
        )

        # Ensure "pred" contains masked elements
        self.assertEqual(
            labelsets["TestTask_slice:f_pred"].numpy().tolist(), [0, 1, -1, -1, -1]
        )
        self.assertEqual(
            labelsets["TestTask_slice:base_pred"].numpy().tolist(), [0, 1, 1, 0, 1]
        )
        self.assertEqual(labelsets["TestTask"].numpy().tolist(), [0, 1, 1, 0, 1])

    def test_convert_to_slice_tasks(self):
        task_name = "TestTask"
        task = create_dummy_task(task_name)

        slice_names = ["slice_a", "slice_b", "slice_c"]
        slice_tasks = convert_to_slice_tasks(task, slice_names)

        slice_task_names = [t.name for t in slice_tasks]
        # Check for original base task
        self.assertIn(task_name, slice_task_names)

        # Check for 2 tasks (pred + ind) per slice, accounting for base slice
        for slice_name in slice_names + ["base"]:
            self.assertIn(f"{task_name}_slice:{slice_name}_pred", slice_task_names)
            self.assertIn(f"{task_name}_slice:{slice_name}_ind", slice_task_names)

        self.assertEqual(len(slice_tasks), 2 * (len(slice_names) + 1) + 1)

        # Test that modules share the same body flow operations
        # NOTE: Use "is" comparison to check object equality
        body_flow = task.op_sequence[:-1]
        ind_and_pred_tasks = [
            t for t in slice_tasks if "_ind" in t.name or "_pred" in t.name
        ]
        for op in body_flow:
            for slice_task in ind_and_pred_tasks:
                self.assertTrue(
                    slice_task.module_pool[op.module_name]
                    is task.module_pool[op.module_name]
                )

        # Test that pred tasks share the same predictor head
        pred_tasks = [t for t in slice_tasks if "_pred" in t.name]
        predictor_head_name = pred_tasks[0].op_sequence[-1].module_name
        shared_predictor_head = pred_tasks[0].module_pool[predictor_head_name]
        for pred_task in pred_tasks[1:]:
            self.assertTrue(
                pred_task.module_pool[predictor_head_name] is shared_predictor_head
            )


def create_dummy_task(task_name):
    # Create dummy task
    module_pool = nn.ModuleDict(
        {"linear1": nn.Linear(2, 10), "linear2": nn.Linear(10, 2)}
    )

    op_sequence = [
        Operation(name="encoder", module_name="linear1", inputs=["_input_"]),
        Operation(name="prediction_head", module_name="linear2", inputs=["encoder"]),
    ]

    task = Task(name=task_name, module_pool=module_pool, op_sequence=op_sequence)
    return task
