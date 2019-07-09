import copy
import unittest

import torch
import torch.nn as nn

from snorkel.classification.data import DictDataLoader, DictDataset
from snorkel.classification.scorer import Scorer
from snorkel.classification.snorkel_classifier import Operation, SnorkelClassifier, Task
from snorkel.classification.training import Trainer

TASK_NAMES = ["task1", "task2"]
trainer_config = {"n_epochs": 2, "progress_bar": False}


def create_dataloader(task_name="task", split="train"):
    X = torch.FloatTensor([[1, 1], [2, 2], [3, 3], [4, 4]])
    Y = torch.LongTensor([1, 1, 2, 2])

    dataset = DictDataset(
        name="dataset", split=split, X_dict={"data": X}, Y_dict={task_name: Y}
    )

    dataloader = DictDataLoader(dataset, batch_size=2)
    return dataloader


def create_task(task_name, module_suffixes=("", "")):
    module1_name = f"linear1{module_suffixes[0]}"
    module2_name = f"linear2{module_suffixes[1]}"

    module_pool = nn.ModuleDict(
        {
            module1_name: nn.Sequential(nn.Linear(2, 10), nn.ReLU()),
            module2_name: nn.Linear(10, 2),
        }
    )

    op1 = Operation(module_name=module1_name, inputs=[("_input_", "data")])
    op2 = Operation(module_name=module2_name, inputs=[(op1.name, 0)])

    task_flow = [op1, op2]

    task = Task(
        name=task_name,
        module_pool=module_pool,
        task_flow=task_flow,
        scorer=Scorer(metrics=["accuracy"]),
    )

    return task


dataloaders = [create_dataloader(task_name) for task_name in TASK_NAMES]
tasks = [
    create_task(TASK_NAMES[0], module_suffixes=["A", "A"]),
    create_task(TASK_NAMES[1], module_suffixes=["A", "B"]),
]


class TrainerTest(unittest.TestCase):
    def test_trainer_onetask(self):
        """Train a single-task model"""
        model = SnorkelClassifier([tasks[0]])
        trainer = Trainer(**trainer_config)
        trainer.train_model(model, [dataloaders[0]])

    def test_trainer_twotask(self):
        """Train a model with overlapping modules and flows"""
        model = SnorkelClassifier(tasks)
        trainer = Trainer(**trainer_config)
        trainer.train_model(model, dataloaders)

    def test_trainer_errors(self):
        model = SnorkelClassifier([tasks[0]])
        dataloader = copy.deepcopy(dataloaders[0])

        # No train split
        trainer = Trainer(**trainer_config)
        dataloader.dataset.split = "valid"
        with self.assertRaises(ValueError):
            trainer.train_model(model, [dataloader])

        # Unused split
        trainer = Trainer(**trainer_config, valid_split="val")
        with self.assertRaises(ValueError):
            trainer.train_model(model, [dataloader])


if __name__ == "__main__":
    unittest.main()
