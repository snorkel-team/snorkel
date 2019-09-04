import unittest
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from snorkel.analysis import Scorer
from snorkel.classification import (
    DictDataLoader,
    DictDataset,
    MultitaskClassifier,
    Operation,
    Task,
    Trainer,
)
from snorkel.utils import set_seed


class ClassifierConvergenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure deterministic runs
        set_seed(123)

        # Create raw data
        cls.N_TRAIN = 1000
        cls.N_VALID = 300

        cls.df_train = create_data(cls.N_TRAIN)
        cls.df_valid = create_data(cls.N_VALID)

    @pytest.mark.complex
    def test_convergence(self):
        """ Test multitask classifier convergence with two tasks."""

        dataloaders = []
        for df, split in [(self.df_train, "train"), (self.df_valid, "valid")]:
            for task_name in ["task1", "task2"]:
                dataloader = create_dataloader(df, split, task_name)
                dataloaders.append(dataloader)

        task1 = create_task("task1", module_suffixes=["A", "A"])
        task2 = create_task("task2", module_suffixes=["A", "B"])
        model = MultitaskClassifier(tasks=[task1, task2])

        # Train
        trainer = Trainer(lr=0.001, n_epochs=10, progress_bar=False)
        trainer.fit(model, dataloaders)
        scores = model.score(dataloaders)

        # Confirm near perfect scores on both tasks
        for idx, task_name in enumerate(["task1", "task2"]):
            self.assertGreater(scores[f"{task_name}/TestData/valid/accuracy"], 0.95)

            # Calculate/check train/val loss
            train_dataset = dataloaders[idx].dataset
            train_loss_output = model.calculate_loss(
                train_dataset.X_dict, train_dataset.Y_dict
            )
            train_loss = train_loss_output[0][task_name].item()
            self.assertLess(train_loss, 0.05)

            val_dataset = dataloaders[2 + idx].dataset
            val_loss_output = model.calculate_loss(
                val_dataset.X_dict, val_dataset.Y_dict
            )
            val_loss = val_loss_output[0][task_name].item()
            self.assertLess(val_loss, 0.05)


def create_data(n: int) -> pd.DataFrame:
    X = np.random.random((n, 2)) * 2 - 1
    Y = (X[:, 0] < X[:, 1] + 0.25).astype(int)

    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": Y})
    return df


def create_dataloader(df: pd.DataFrame, split: str, task_name: str) -> DictDataLoader:
    dataset = DictDataset(
        name="TestData",
        split=split,
        X_dict={
            "coordinates": torch.stack(
                (torch.tensor(df["x1"]), torch.tensor(df["x2"])), dim=1
            )
        },
        Y_dict={task_name: torch.tensor(df["y"], dtype=torch.long)},
    )

    dataloader = DictDataLoader(
        dataset=dataset, batch_size=4, shuffle=(dataset.split == "train")
    )
    return dataloader


def create_task(task_name: str, module_suffixes: List[str]) -> Task:
    module1_name = f"linear1{module_suffixes[0]}"
    module2_name = f"linear2{module_suffixes[1]}"

    module_pool = nn.ModuleDict(
        {
            module1_name: nn.Sequential(nn.Linear(2, 20), nn.ReLU()),
            module2_name: nn.Linear(20, 2),
        }
    )

    op1 = Operation(module_name=module1_name, inputs=[("_input_", "coordinates")])
    op2 = Operation(module_name=module2_name, inputs=[op1.name])

    op_sequence = [op1, op2]

    task = Task(
        name=task_name,
        module_pool=module_pool,
        op_sequence=op_sequence,
        scorer=Scorer(metrics=["accuracy"]),
    )

    return task


if __name__ == "__main__":
    unittest.main()
