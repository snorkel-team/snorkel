import unittest

import pandas as pd
import torch
import torch.nn as nn

from snorkel.classification.data import DictDataLoader, DictDataset
from snorkel.classification.scorer import Scorer
from snorkel.classification.snorkel_classifier import Operation, Task
from snorkel.slicing.apply import PandasSFApplier
from snorkel.slicing.sf import slicing_function
from snorkel.slicing.utils import add_slice_labels, convert_to_slice_tasks
from snorkel.types import DataPoint


@slicing_function()
def f(x: DataPoint) -> int:
    return x.val < 0.25


class UtilsTest(unittest.TestCase):
    def test_add_slice_labels(self):
        # Create dummy data
        x = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        y = torch.Tensor([2, 1, 1, 2, 1]).long()
        dataset = DictDataset(
            name="TestData", split="train", X_dict={"data": x}, Y_dict={"TestTask": y}
        )

        # Ensure that we start with 1 labelset
        self.assertEqual(len(dataset.Y_dict), 1)

        # Apply SFs with PandasSFApplier
        df = pd.DataFrame({"val": x, "y": y})
        slicing_functions = [f]
        slice_names = [sf.name for sf in slicing_functions]
        applier = PandasSFApplier(slicing_functions)
        S = applier.apply(df)

        dataloader = DictDataLoader(dataset)

        dummy_task = create_dummy_task(task_name="TestTask")
        add_slice_labels(dataloader, dummy_task, S, slice_names)

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
            labelsets["TestTask_slice:f_ind"].numpy().tolist(), [1, 1, 2, 2, 2]
        )
        self.assertEqual(
            labelsets["TestTask_slice:base_ind"].numpy().tolist(), [1, 1, 1, 1, 1]
        )

        # Ensure "pred" contains masked elements
        self.assertEqual(
            labelsets["TestTask_slice:f_pred"].numpy().tolist(), [2, 1, 0, 0, 0]
        )
        self.assertEqual(
            labelsets["TestTask_slice:base_pred"].numpy().tolist(), [2, 1, 1, 2, 1]
        )
        self.assertEqual(labelsets["TestTask"].numpy().tolist(), [2, 1, 1, 2, 1])

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


def create_dummy_task(task_name):
    # Create dummy task
    module_pool = nn.ModuleDict({"linear1": nn.Linear(1, 1)})

    task_flow = [
        Operation(
            name="the_first_layer", module_name="linear1", inputs=[("_input_", 0)]
        )
    ]

    task = Task(
        name=task_name,
        module_pool=module_pool,
        task_flow=task_flow,
        scorer=Scorer(metrics=["accuracy"]),
    )
    return task
