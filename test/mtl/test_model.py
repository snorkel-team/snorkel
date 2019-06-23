import unittest
from functools import partial

import torch.nn as nn

from snorkel.mtl.model import MultitaskModel
from snorkel.mtl.modules.utils import ce_loss, softmax
from snorkel.mtl.scorer import Scorer
from snorkel.mtl.task import Operation, Task


class TaskTest(unittest.TestCase):
    def create_task(self, task_name, module_suffixes):
        module1_name = f"linear1{module_suffixes[0]}"
        module2_name = f"linear2{module_suffixes[1]}"

        module_pool = nn.ModuleDict(
            {module1_name: nn.Linear(2, 10), module2_name: nn.Linear(10, 2)}
        )

        op1 = Operation(module_name=module1_name, inputs=[("_input_", "coordinates")])
        op2 = Operation(module_name=module2_name, inputs=[(op1.name, 0)])

        task_flow = [op1, op2]

        task = Task(
            name=task_name,
            module_pool=module_pool,
            task_flow=task_flow,
            loss_func=partial(ce_loss, op2.name),
            output_func=partial(softmax, op2.name),
            scorer=Scorer(metrics=["accuracy"]),
        )

        return task

    def test_onetask_model(self):
        task1 = self.create_task("task1", module_suffixes=["A", "A"])
        model = MultitaskModel(tasks=[task1])
        self.assertEqual(len(model.task_names), 1)
        self.assertEqual(len(model.task_flows), 1)
        self.assertEqual(len(model.module_pool), 2)

    def test_twotask_all_overlap_model(self):
        """Add two tasks with identical modules and flows"""
        task1 = self.create_task("task1", module_suffixes=["A", "A"])
        task2 = self.create_task("task2", module_suffixes=["A", "A"])
        model = MultitaskModel(tasks=[task1, task2])
        self.assertEqual(len(model.task_names), 2)
        self.assertEqual(len(model.task_flows), 2)
        self.assertEqual(len(model.module_pool), 2)

    def test_twotask_none_overlap_model(self):
        """Add two tasks with totally separate modules and flows"""
        task1 = self.create_task("task1", module_suffixes=["A", "A"])
        task2 = self.create_task("task2", module_suffixes=["B", "B"])
        model = MultitaskModel(tasks=[task1, task2])
        self.assertEqual(len(model.task_names), 2)
        self.assertEqual(len(model.task_flows), 2)
        self.assertEqual(len(model.module_pool), 4)

    def test_twotask_partial_overlap_model(self):
        """Add two tasks with overlapping modules and flows"""
        task1 = self.create_task("task1", module_suffixes=["A", "A"])
        task2 = self.create_task("task2", module_suffixes=["A", "B"])
        model = MultitaskModel(tasks=[task1, task2])
        self.assertEqual(len(model.task_names), 2)
        self.assertEqual(len(model.task_flows), 2)
        self.assertEqual(len(model.module_pool), 3)
