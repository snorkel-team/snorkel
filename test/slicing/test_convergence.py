import random
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
from snorkel.slicing import (
    PandasSFApplier,
    add_slice_labels,
    convert_to_slice_tasks,
    slicing_function,
)
from snorkel.types import DataPoint


# Define SFs specifying points inside a circle
@slicing_function()
def f(x: DataPoint) -> int:
    # targets ~7% of the data
    radius = 0.3
    h, k = (-0.15, -0.3)  # center
    return np.sqrt((x.x1 - h) ** 2 + (x.x2 - k) ** 2) < radius


@slicing_function()
def g(x: DataPoint) -> int:
    # targets ~6% of the data
    radius = 0.3
    h, k = (0.25, 0.0)  # center
    return np.sqrt((x.x1 - h) ** 2 + (x.x2 - k) ** 2) < radius


@slicing_function()
def h(x: DataPoint) -> int:
    # targets ~27% of the data
    radius = 0.6
    h, k = (0.25, 0.0)  # center
    return np.sqrt((x.x1 - h) ** 2 + (x.x2 - k) ** 2) < radius


class SlicingConvergenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure deterministic runs
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        # Create raw data
        cls.N_TRAIN = 1500
        cls.N_VALID = 300

        cls.df_train = create_data(cls.N_TRAIN)
        cls.df_valid = create_data(cls.N_VALID)

    @pytest.mark.complex
    def test_convergence(self):
        """ Test slicing convergence with 1 slice task that represents ~25% of
        the data. """

        dataloaders = []
        for df, split in [(self.df_train, "train"), (self.df_valid, "valid")]:
            dataloader = create_dataloader(df, split)
            dataloaders.append(dataloader)

        base_task = create_task("task", module_suffixes=["A", "B"])

        # Apply SFs
        slicing_functions = [h]  # high coverage slice
        slice_names = [sf.name for sf in slicing_functions]
        applier = PandasSFApplier(slicing_functions)
        S_train = applier.apply(self.df_train, progress_bar=False)
        S_valid = applier.apply(self.df_valid, progress_bar=False)

        self.assertEqual(S_train.shape, (self.N_TRAIN,))
        self.assertEqual(S_valid.shape, (self.N_VALID,))
        self.assertIn("h", S_train.dtype.names)

        # Add slice labels
        add_slice_labels(dataloaders[0], base_task, S_train)
        add_slice_labels(dataloaders[1], base_task, S_valid)

        # Convert to slice tasks
        tasks = convert_to_slice_tasks(base_task, slice_names)
        model = MultitaskClassifier(tasks=tasks)

        # Train
        trainer = Trainer(lr=0.001, n_epochs=50, progress_bar=False)
        trainer.fit(model, dataloaders)
        scores = model.score(dataloaders)

        # Confirm near perfect scores
        self.assertGreater(scores["task/TestData/valid/accuracy"], 0.95)
        self.assertGreater(scores["task_slice:h_pred/TestData/valid/accuracy"], 0.95)
        self.assertGreater(scores["task_slice:h_ind/TestData/valid/f1"], 0.95)

        # Calculate/check train/val loss
        train_dataset = dataloaders[0].dataset
        train_loss_output = model.calculate_loss(
            train_dataset.X_dict, train_dataset.Y_dict
        )
        train_loss = train_loss_output[0]["task"].item()
        self.assertLess(train_loss, 0.1)

        val_dataset = dataloaders[1].dataset
        val_loss_output = model.calculate_loss(val_dataset.X_dict, val_dataset.Y_dict)
        val_loss = val_loss_output[0]["task"].item()
        self.assertLess(val_loss, 0.1)

    @pytest.mark.complex
    def test_performance(self):
        """ Test slicing performance with 2 corresponding slice tasks that
        represent roughly <10% of the data. """

        dataloaders = []
        for df, split in [(self.df_train, "train"), (self.df_valid, "valid")]:
            dataloader = create_dataloader(df, split)
            dataloaders.append(dataloader)

        base_task = create_task("task", module_suffixes=["A", "B"])

        # Apply SFs
        slicing_functions = [f, g]  # low-coverage slices
        slice_names = [sf.name for sf in slicing_functions]
        applier = PandasSFApplier(slicing_functions)
        S_train = applier.apply(self.df_train, progress_bar=False)
        S_valid = applier.apply(self.df_valid, progress_bar=False)

        # Add slice labels
        add_slice_labels(dataloaders[0], base_task, S_train)
        add_slice_labels(dataloaders[1], base_task, S_valid)

        # Convert to slice tasks
        tasks = convert_to_slice_tasks(base_task, slice_names)
        model = MultitaskClassifier(tasks=tasks)

        # Train
        # NOTE: Needs more epochs to convergence with more heads
        trainer = Trainer(lr=0.001, n_epochs=65, progress_bar=False)
        trainer.fit(model, dataloaders)
        scores = model.score(dataloaders)

        # Confirm reasonably high slice scores
        # Check train scores
        self.assertGreater(scores["task/TestData/train/f1"], 0.9)
        self.assertGreater(scores["task_slice:f_pred/TestData/train/f1"], 0.9)
        self.assertGreater(scores["task_slice:f_ind/TestData/train/f1"], 0.9)
        self.assertGreater(scores["task_slice:g_pred/TestData/train/f1"], 0.9)
        self.assertGreater(scores["task_slice:g_ind/TestData/train/f1"], 0.9)
        self.assertGreater(scores["task_slice:base_pred/TestData/train/f1"], 0.9)
        self.assertEqual(scores["task_slice:base_ind/TestData/train/f1"], 1.0)

        # Check valid scores
        self.assertGreater(scores["task/TestData/valid/f1"], 0.9)
        self.assertGreater(scores["task_slice:f_pred/TestData/valid/f1"], 0.9)
        self.assertGreater(scores["task_slice:f_ind/TestData/valid/f1"], 0.9)
        self.assertGreater(scores["task_slice:g_pred/TestData/valid/f1"], 0.9)
        self.assertGreater(scores["task_slice:g_ind/TestData/valid/f1"], 0.9)
        self.assertGreater(scores["task_slice:base_pred/TestData/valid/f1"], 0.9)
        # base_ind is trivial: all labels are positive
        self.assertEqual(scores["task_slice:base_ind/TestData/valid/f1"], 1.0)


def create_data(n: int) -> pd.DataFrame:
    X = np.random.random((n, 2)) * 2 - 1
    Y = (X[:, 0] < X[:, 1] + 0.25).astype(int)

    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": Y})
    return df


def create_dataloader(df: pd.DataFrame, split: str) -> DictDataLoader:
    dataset = DictDataset(
        name="TestData",
        split=split,
        X_dict={
            "coordinates": torch.stack(
                (torch.tensor(df["x1"]), torch.tensor(df["x2"])), dim=1
            )
        },
        Y_dict={"task": torch.tensor(df["y"], dtype=torch.long)},
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
        scorer=Scorer(metrics=["f1", "accuracy"]),
    )

    return task


if __name__ == "__main__":
    unittest.main()
