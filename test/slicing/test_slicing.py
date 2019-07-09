import unittest

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from snorkel.classification.data import DictDataLoader, DictDataset
from snorkel.classification.scorer import Scorer
from snorkel.classification.snorkel_classifier import Operation, SnorkelClassifier, Task
from snorkel.classification.training import Trainer
from snorkel.slicing.apply import PandasSFApplier
from snorkel.slicing.sf import slicing_function
from snorkel.slicing.utils import add_slice_labels, convert_to_slice_tasks
from snorkel.types import DataPoint

SEED = 123


@slicing_function()
def f(x: DataPoint) -> int:
    return 1 if x.x1 > x.x2 + 0.5 else 0


@slicing_function()
def g(x: DataPoint) -> int:
    return 1 if x.x1 > x.x2 + 0.3 else 0


N_TRAIN = 1000
N_VALID = 100


class SlicingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_config = {"n_epochs": 3, "progress_bar": False}

    @pytest.mark.complex
    def test_slicing(self):
        """Define two slices for task1 and no slices for task2"""
        df_train = create_data(N_TRAIN)
        df_valid = create_data(N_VALID)

        dataloaders = []
        for df, split in [(df_train, "train"), (df_valid, "valid")]:
            dataloader = create_dataloader(df, split)
            dataloaders.append(dataloader)

        task1 = create_task("task1", module_suffixes=["A", "A"])
        task2 = create_task("task2", module_suffixes=["A", "B"])

        # Apply SFs
        slicing_functions = [f, g]
        slice_names = [sf.name for sf in slicing_functions]
        applier = PandasSFApplier(slicing_functions)
        S_train = applier.apply(df_train)
        S_valid = applier.apply(df_valid)

        self.assertEqual(S_train.shape, (N_TRAIN, len(slicing_functions)))
        self.assertEqual(S_valid.shape, (N_VALID, len(slicing_functions)))

        # Add slice labels
        add_slice_labels(dataloaders[0], task1, S_train, slice_names)
        add_slice_labels(dataloaders[1], task1, S_valid, slice_names)

        # Convert to slice tasks
        task1_tasks = convert_to_slice_tasks(task1, slice_names)
        tasks = task1_tasks + [task2]
        model = SnorkelClassifier(tasks=tasks)

        # Train
        trainer = Trainer(**self.trainer_config)
        trainer.train_model(model, dataloaders)
        model.score(dataloaders)


def create_data(n):

    X = np.random.random((n, 2)) * 2 - 1
    Y = np.zeros((n, 2))
    Y[:, 0] = (X[:, 0] > X[:, 1] + 0.5).astype(int) + 1
    Y[:, 1] = (X[:, 0] > X[:, 1] + 0.25).astype(int) + 1

    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y1": Y[:, 0], "y2": Y[:, 1]})
    return df


def create_dataloader(df, split):
    Y_dict = {}

    Y_dict[f"task1"] = torch.LongTensor(df["y1"])
    Y_dict[f"task2"] = torch.LongTensor(df["y2"])

    dataset = DictDataset(
        name="TestData",
        split=split,
        X_dict={
            "coordinates": torch.stack(
                (torch.Tensor(df["x1"]), torch.Tensor(df["x2"])), dim=1
            )
        },
        Y_dict=Y_dict,
    )

    dataloader = DictDataLoader(
        dataset=dataset, batch_size=4, shuffle=(dataset.split == "train")
    )
    return dataloader


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
        scorer=Scorer(metrics=["accuracy"]),
    )

    return task


if __name__ == "__main__":
    unittest.main()
