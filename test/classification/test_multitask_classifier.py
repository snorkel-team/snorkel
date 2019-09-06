import os
import random
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from snorkel.classification import (
    DictDataLoader,
    DictDataset,
    MultitaskClassifier,
    Operation,
    Task,
)

NUM_EXAMPLES = 10
BATCH_SIZE = 2


class ClassifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task1 = create_task("task1", module_suffixes=["A", "A"])
        cls.task2 = create_task("task2", module_suffixes=["B", "B"])
        cls.dataloader = create_dataloader("task1")

    def setUp(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

    def test_onetask_model(self):
        model = MultitaskClassifier(tasks=[self.task1])
        self.assertEqual(len(model.task_names), 1)
        self.assertEqual(len(model.op_sequences), 1)
        self.assertEqual(len(model.module_pool), 2)

    def test_twotask_none_overlap_model(self):
        """Add two tasks with totally separate modules and flows"""
        model = MultitaskClassifier(tasks=[self.task1, self.task2])
        self.assertEqual(len(model.task_names), 2)
        self.assertEqual(len(model.op_sequences), 2)
        self.assertEqual(len(model.module_pool), 4)

    def test_twotask_all_overlap_model(self):
        """Add two tasks with identical modules and flows"""
        task1 = create_task("task1", module_suffixes=["A", "A"])
        task2 = create_task("task2", module_suffixes=["A", "A"])
        model = MultitaskClassifier(tasks=[task1, task2])
        self.assertEqual(len(model.task_names), 2)
        self.assertEqual(len(model.op_sequences), 2)
        self.assertEqual(len(model.module_pool), 2)

    def test_twotask_partial_overlap_model(self):
        """Add two tasks with overlapping modules and flows"""
        task1 = create_task("task1", module_suffixes=["A", "A"])
        task2 = create_task("task2", module_suffixes=["A", "B"])
        model = MultitaskClassifier(tasks=[task1, task2])
        self.assertEqual(len(model.task_names), 2)
        self.assertEqual(len(model.op_sequences), 2)
        self.assertEqual(len(model.module_pool), 3)

    def test_bad_tasks(self):
        with self.assertRaisesRegex(ValueError, "Found duplicate task"):
            MultitaskClassifier(tasks=[self.task1, self.task1])
        with self.assertRaisesRegex(ValueError, "Unrecognized task type"):
            MultitaskClassifier(tasks=[self.task1, {"fake_task": 42}])
        with self.assertRaisesRegex(ValueError, "Unsuccessful operation"):
            task1 = create_task("task1")
            task1.op_sequence[0].inputs[0] = (0, 0)
            model = MultitaskClassifier(tasks=[task1])
            X_dict = self.dataloader.dataset.X_dict
            model.forward(X_dict, [task1.name])

    def test_no_data_parallel(self):
        model = MultitaskClassifier(tasks=[self.task1, self.task2], dataparallel=False)
        self.assertEqual(len(model.task_names), 2)
        self.assertIsInstance(model.module_pool["linear1A"], nn.Module)

    def test_no_input_spec(self):
        # Confirm model doesn't break when a module does not specify specific inputs
        dataset = create_dataloader("task", shuffle=False).dataset
        task = Task(
            name="task",
            module_pool=nn.ModuleDict({"identity": nn.Identity()}),
            op_sequence=[Operation("identity", [])],
        )
        model = MultitaskClassifier(tasks=[task], dataparallel=False)
        outputs = model.forward(dataset.X_dict, ["task"])
        self.assertIn("_input_", outputs)

    def test_predict(self):
        model = MultitaskClassifier([self.task1])
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
        dataset = create_dataloader("task1", shuffle=False).dataset
        dataset.Y_dict["task1"] = torch.full_like(dataset.Y_dict["task1"], -1)
        model = MultitaskClassifier([self.task1])
        loss_dict, count_dict = model.calculate_loss(dataset.X_dict, dataset.Y_dict)
        self.assertFalse(loss_dict)
        self.assertFalse(count_dict)

    def test_partially_empty_batch(self):
        dataset = create_dataloader("task1", shuffle=False).dataset
        dataset.Y_dict["task1"][0] = -1
        model = MultitaskClassifier([self.task1])
        loss_dict, count_dict = model.calculate_loss(dataset.X_dict, dataset.Y_dict)
        self.assertEqual(count_dict["task1"], 9)

    def test_remapped_labels(self):
        # Test additional label keys in the Y_dict
        # Without remapping, model should ignore them
        task_name = self.task1.name
        X = torch.FloatTensor([[i, i] for i in range(NUM_EXAMPLES)])
        Y = torch.ones(NUM_EXAMPLES).long()

        Y_dict = {task_name: Y, "other_task": Y}
        dataset = DictDataset(
            name="dataset", split="train", X_dict={"data": X}, Y_dict=Y_dict
        )
        dataloader = DictDataLoader(dataset, batch_size=BATCH_SIZE)

        model = MultitaskClassifier([self.task1])
        loss_dict, count_dict = model.calculate_loss(dataset.X_dict, dataset.Y_dict)
        self.assertIn("task1", loss_dict)

        # Test setting without remapping
        results = model.predict(dataloader)
        self.assertIn("task1", results["golds"])
        self.assertNotIn("other_task", results["golds"])
        scores = model.score([dataloader])
        self.assertIn("task1/dataset/train/accuracy", scores)
        self.assertNotIn("other_task/dataset/train/accuracy", scores)

        # Test remapped labelsets
        results = model.predict(dataloader, remap_labels={"other_task": task_name})
        self.assertIn("task1", results["golds"])
        self.assertIn("other_task", results["golds"])
        results = model.score([dataloader], remap_labels={"other_task": task_name})
        self.assertIn("task1/dataset/train/accuracy", results)
        self.assertIn("other_task/dataset/train/accuracy", results)

    def test_score(self):
        model = MultitaskClassifier([self.task1])
        metrics = model.score([self.dataloader])
        # deterministic random tie breaking alternates predicted labels
        self.assertEqual(metrics["task1/dataset/train/accuracy"], 0.4)

        # test dataframe format
        metrics_df = model.score([self.dataloader], as_dataframe=True)
        self.assertTrue(isinstance(metrics_df, pd.DataFrame))
        self.assertEqual(metrics_df.at[0, "score"], 0.4)

    def test_score_shuffled(self):
        # Test scoring with a shuffled dataset

        class SimpleVoter(nn.Module):
            def forward(self, x):
                """Set class 0 to -1 if x and 1 otherwise"""
                mask = x % 2 == 0
                out = torch.zeros(x.shape[0], 2)
                out[mask, 0] = 1  # class 0
                out[~mask, 1] = 1  # class 1
                return out

        # Create model
        task_name = "VotingTask"
        module_name = "simple_voter"
        module_pool = nn.ModuleDict({module_name: SimpleVoter()})
        op0 = Operation(
            module_name=module_name, inputs=[("_input_", "data")], name="op0"
        )
        op_sequence = [op0]
        task = Task(name=task_name, module_pool=module_pool, op_sequence=op_sequence)
        model = MultitaskClassifier([task])

        # Create dataset
        y_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        x_list = [i for i in range(len(y_list))]
        Y = torch.LongTensor(y_list * 100)
        X = torch.FloatTensor(x_list * 100)
        dataset = DictDataset(
            name="dataset", split="train", X_dict={"data": X}, Y_dict={task_name: Y}
        )

        # Create dataloaders
        dataloader = DictDataLoader(dataset, batch_size=2, shuffle=False)
        scores = model.score([dataloader])

        self.assertEqual(scores["VotingTask/dataset/train/accuracy"], 0.6)

        dataloader_shuffled = DictDataLoader(dataset, batch_size=2, shuffle=True)
        scores_shuffled = model.score([dataloader_shuffled])
        self.assertEqual(scores_shuffled["VotingTask/dataset/train/accuracy"], 0.6)

    def test_save_load(self):
        fd, checkpoint_path = tempfile.mkstemp()

        task1 = create_task("task1")
        task2 = create_task("task2")
        # Make task2's second linear layer have different weights
        task2.module_pool["linear2"] = nn.Linear(2, 2)

        model = MultitaskClassifier([task1])
        self.assertTrue(
            torch.eq(
                task1.module_pool["linear2"].weight,
                model.module_pool["linear2"].module.weight,
            ).all()
        )
        model.save(checkpoint_path)
        model = MultitaskClassifier([task2])
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
    Y = torch.ones(NUM_EXAMPLES).long()

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
    op1 = Operation(module_name=module2_name, inputs=[op0.name], name="op1")

    op_sequence = [op0, op1]

    task = Task(name=task_name, module_pool=module_pool, op_sequence=op_sequence)

    return task


if __name__ == "__main__":
    unittest.main()
