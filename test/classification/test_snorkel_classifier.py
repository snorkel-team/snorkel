import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from snorkel.classification.data import DictDataLoader, DictDataset
from snorkel.classification.scorer import Scorer
from snorkel.classification.snorkel_classifier import Operation, SnorkelClassifier, Task

NUM_EXAMPLES = 10
BATCH_SIZE = 2


class ClassifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task1 = create_task("task1", module_suffixes=["A", "A"])
        cls.task2 = create_task("task2", module_suffixes=["B", "B"])
        cls.dataloader = create_dataloader("task1")

    def test_onetask_model(self):
        model = SnorkelClassifier(tasks=[self.task1])
        self.assertEqual(len(model.task_names), 1)
        self.assertEqual(len(model.task_flows), 1)
        self.assertEqual(len(model.module_pool), 2)

    def test_twotask_none_overlap_model(self):
        """Add two tasks with totally separate modules and flows"""
        model = SnorkelClassifier(tasks=[self.task1, self.task2])
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
            SnorkelClassifier(tasks=[self.task1, self.task1])
        with self.assertRaisesRegex(ValueError, "Unrecognized task type"):
            SnorkelClassifier(tasks=[self.task1, {"fake_task": 42}])
        with self.assertRaisesRegex(ValueError, "Unsuccessful operation"):
            task1 = create_task("task1")
            task1.task_flow[0].inputs[0] = (0, 0)
            model = SnorkelClassifier(tasks=[task1])
            X_dict = self.dataloader.dataset.X_dict
            model.forward(X_dict, [task1.name])

    def test_predict(self):
        model = SnorkelClassifier([self.task1])
        results = model.predict(self.dataloader)
        self.assertEqual(sorted(list(results.keys())), ["golds", "probs"])
        np.testing.assert_array_equal(
            results["golds"]["task1"], self.dataloader.dataset.Y_dict["task1"].numpy()
        )
        np.testing.assert_array_equal(
            results["probs"]["task1"], np.ones((NUM_EXAMPLES, 2)) * 0.5
        )

        results = model.predict(self.dataloader, return_preds=True)
        self.assertEqual(sorted(list(results.keys())), ["golds", "preds", "probs"])
        # deterministic random tie breaking alternates predicted labels
        np.testing.assert_array_equal(
            results["preds"]["task1"],
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
        )

    def test_empty_batch(self):
        # Make the first BATCH_SIZE labels -1 so that one batch will have no labels
        dataset = create_dataloader("task1", shuffle=False).dataset
        for i in range(BATCH_SIZE):
            dataset.Y_dict["task1"][i] = -1
        model = SnorkelClassifier([self.task1])
        loss_dict, count_dict = model.calculate_loss(dataset.X_dict, dataset.Y_dict)
        self.assertEqual(count_dict["task1"], NUM_EXAMPLES - BATCH_SIZE)

    def test_score(self):
        model = SnorkelClassifier([self.task1])
        metrics = model.score([self.dataloader])
        # deterministic random tie breaking alternates predicted labels
        self.assertEqual(metrics["task1/dataset/train/accuracy"], 0.4)

    def test_save_load(self):
        fd, checkpoint_path = tempfile.mkstemp()

        task1 = create_task("task1")
        task2 = create_task("task2")
        # Make task2's second linear layer have different weights
        task2.module_pool["linear2"] = nn.Linear(2, 2)

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


def create_dataloader(task_name="task", split="train", **kwargs):
    X = torch.FloatTensor([[i, i] for i in range(NUM_EXAMPLES)])
    Y = torch.ones(NUM_EXAMPLES, 1).long()

    dataset = DictDataset(
        name="dataset", split=split, X_dict={"data": X}, Y_dict={task_name: Y}
    )

    dataloader = DictDataLoader(dataset, batch_size=BATCH_SIZE, **kwargs)
    return dataloader


def create_task(task_name, module_suffixes=("", "")):
    module1_name = f"linear1{module_suffixes[0]}"
    module2_name = f"linear2{module_suffixes[1]}"

    linear1 = nn.Linear(2, 2)
    linear1.weight.data.copy_(torch.eye(2))
    linear1.bias.data.copy_(torch.zeros((2,)))

    linear2 = nn.Linear(2, 2)
    linear2.weight.data.copy_(torch.eye(2))
    linear2.bias.data.copy_(torch.zeros((2,)))

    module_pool = nn.ModuleDict(
        {module1_name: nn.Sequential(linear1, nn.ReLU()), module2_name: linear2}
    )

    op0 = Operation(module_name=module1_name, inputs=[("_input_", "data")], name="op0")
    op1 = Operation(module_name=module2_name, inputs=[(op0.name, 0)], name="op1")

    task_flow = [op0, op1]

    task = Task(
        name=task_name,
        module_pool=module_pool,
        task_flow=task_flow,
    )

    return task


if __name__ == "__main__":
    unittest.main()
