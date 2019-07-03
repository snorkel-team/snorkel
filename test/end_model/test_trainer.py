import unittest
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from snorkel.end_model.data import SnorkelDataLoader, SnorkelDataset
from snorkel.end_model.model import MultitaskModel
from snorkel.end_model.modules.utils import ce_loss, softmax
from snorkel.end_model.scorer import Scorer
from snorkel.end_model.task import Operation, Task
from snorkel.end_model.trainer import Trainer

SEED = 123


class TrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_config = {"n_epochs": 2, "progress_bar": False}

    def test_trainer_onetask(self):
        """Train a single-task model"""
        task1 = create_task("task1", module_suffixes=["A", "A"])
        model = MultitaskModel(tasks=[task1])
        dataloaders = create_dataloaders(num_tasks=1)
        trainer = Trainer(**self.trainer_config)
        trainer.train_model(model, dataloaders)

    def test_trainer_twotask(self):
        """Train a model with overlapping modules and flows"""
        task1 = create_task("task1", module_suffixes=["A", "A"])
        task2 = create_task("task2", module_suffixes=["A", "B"])
        model = MultitaskModel(tasks=[task1, task2])
        dataloaders = create_dataloaders(num_tasks=2)
        trainer = Trainer(**self.trainer_config)
        trainer.train_model(model, dataloaders)


def create_dataloaders(num_tasks=1):
    n = 1200

    X = np.random.random((n, 2)) * 2 - 1
    Y = np.zeros((n, 2))
    Y[:, 0] = (X[:, 0] > X[:, 1] + 0.5).astype(int) + 1
    Y[:, 1] = (X[:, 0] > X[:, 1] + 0.25).astype(int) + 1

    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.long)

    Xs = [X[:1000], X[1000:1100], X[1100:]]
    Ys = [Y[:1000], Y[1000:1100], Y[1100:]]

    dataloaders = []
    splits = ["train", "valid", "test"]
    for X_split, Y_split, split in zip(Xs, Ys, splits):

        Y_dict = {"task1_labels": Y_split[:, 0]}
        task_to_label_dict = {"task1": "task1_labels"}
        if num_tasks == 2:
            Y_dict["task2_labels"] = Y_split[:, 1]
            task_to_label_dict["task2"] = "task2_labels"

        dataset = SnorkelDataset(
            name="dataset", split=split, X_dict={"coordinates": X_split}, Y_dict=Y_dict
        )

        dataloader = SnorkelDataLoader(
            task_to_label_dict=task_to_label_dict,
            dataset=dataset,
            batch_size=4,
            shuffle=(dataset.split == "train"),
        )
        dataloaders.append(dataloader)
    return dataloaders


def create_task(task_name, module_suffixes):
    module1_name = f"linear1{module_suffixes[0]}"
    module2_name = f"linear2{module_suffixes[1]}"

    module_pool = nn.ModuleDict(
        {
            module1_name: nn.Sequential(nn.Linear(2, 10), nn.ReLU()),
            module2_name: nn.Linear(10, 2),
        }
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


if __name__ == "__main__":
    unittest.main()
