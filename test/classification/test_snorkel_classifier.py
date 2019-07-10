import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from snorkel.classification.scorer import Scorer
from snorkel.classification.snorkel_classifier import Operation, SnorkelClassifier, Task
from snorkel.classification.data import DictDataset, DictDataLoader

NUM_EXAMPLES = 10
BATCH_SIZE = 2


def create_dataloader(task_name="task", split="train"):
    X = torch.FloatTensor([[i, i] for i in range(NUM_EXAMPLES)])
    Y = torch.ones(NUM_EXAMPLES, 1).long()

    dataset = DictDataset(
        name="dataset", split=split, X_dict={"data": X}, Y_dict={task_name: Y}
    )

    dataloader = DictDataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader


def create_task(task_name, module_suffixes=("", "")):
    module1_name = f"linear1{module_suffixes[0]}"
    module2_name = f"linear2{module_suffixes[1]}"

    module_pool = nn.ModuleDict(
        {
            module1_name: nn.Sequential(nn.Linear(2, 4), nn.ReLU()),
            module2_name: nn.Linear(4, 2),
        }
    )

    op0 = Operation(module_name=module1_name, inputs=[("_input_", "data")], name="op0")
    op1 = Operation(module_name=module2_name, inputs=[(op0.name, 0)], name="op1")

    task_flow = [op0, op1]

    task = Task(
        name=task_name,
        module_pool=module_pool,
        task_flow=task_flow,
        scorer=Scorer(metrics=["accuracy"]),
    )

    return task


task1 = create_task("task1", module_suffixes=["A", "A"])
task2 = create_task("task2", module_suffixes=["B", "B"])
dataloader = create_dataloader("task1")


class ClassifierTest(unittest.TestCase):
    def test_onetask_model(self):
        model = SnorkelClassifier(tasks=[task1])
        self.assertEqual(len(model.task_names), 1)
        self.assertEqual(len(model.task_flows), 1)
        self.assertEqual(len(model.module_pool), 2)

    def test_twotask_none_overlap_model(self):
        """Add two tasks with totally separate modules and flows"""
        model = SnorkelClassifier(tasks=[task1, task2])
        self.assertEqual(len(model.task_names), 2)
        self.assertEqual(len(model.task_flows), 2)
        self.assertEqual(len(model.module_pool), 4)

    def test_twotask_all_overlap_model(self):
        """Add two tasks with identical modules and flows"""
        task1 = create_task("task1", module_suffixes=["A", "A"])
        task2 = create_task("task2", module_suffixes=["A", "A"])
        model = SnorkelClassifier(tasks=[task1, task2])
        self.assertEqual(len(model.task_names), 2)
        self.assertEqual(len(model.task_flows), 2)
        self.assertEqual(len(model.module_pool), 2)

    def test_twotask_partial_overlap_model(self):
        """Add two tasks with overlapping modules and flows"""
        task1 = create_task("task1", module_suffixes=["A", "A"])
        task2 = create_task("task2", module_suffixes=["A", "B"])
        model = SnorkelClassifier(tasks=[task1, task2])
        self.assertEqual(len(model.task_names), 2)
        self.assertEqual(len(model.task_flows), 2)
        self.assertEqual(len(model.module_pool), 3)

    def test_bad_tasks(self):
        with self.assertRaisesRegex(ValueError, "Found duplicate task"):
            SnorkelClassifier(tasks=[task1, task1])
        with self.assertRaisesRegex(ValueError, "Unrecognized task type"):
            SnorkelClassifier(tasks=[task1, {"fake_task": 42}])

    def test_predict(self):
        model = SnorkelClassifier([task1])
        results = model.predict(dataloader)
        self.assertEqual(sorted(list(results.keys())), ["golds", "probs"])
        np.testing.assert_array_equal(results["golds"]["task1"], dataloader.dataset.Y_dict["task1"].numpy())
        self.assertEqual(results["probs"]["task1"].shape, (NUM_EXAMPLES, 2))

        results = model.predict(dataloader, return_preds=True)
        self.assertEqual(sorted(list(results.keys())), ["golds", "preds", "probs"])
        self.assertEqual(results["preds"]["task1"].shape, (NUM_EXAMPLES,))

    def test_score(self):
        model = SnorkelClassifier([task1])
        metrics = model.score([dataloader])
        self.assertIsInstance(metrics["task1/dataset/train/accuracy"], float)

    def test_save_load(self):
        fd, checkpoint_path = tempfile.mkstemp()

        task1 = create_task("task1")
        task2 = create_task("task2")

        model = SnorkelClassifier([task1])
        self.assertTrue(
            torch.eq(
                task1.module_pool["linear2"].weight,
                model.module_pool["linear2"].module.weight,
            ).all()
        )
        model.save(checkpoint_path)
        model = SnorkelClassifier([task2])
        self.assertFalse(
            torch.eq(
                task1.module_pool["linear2"].weight,
                model.module_pool["linear2"].module.weight,
            ).all()
        )
        model.load(checkpoint_path)
        self.assertTrue(
            torch.eq(
                task1.module_pool["linear2"].weight,
                model.module_pool["linear2"].module.weight,
            ).all()
        )

        os.close(fd)


if __name__ == "__main__":
    unittest.main()
