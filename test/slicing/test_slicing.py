import random
import unittest
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from snorkel.slicing.apply import PandasSFApplier
from snorkel.slicing.sf import slicing_function 
from snorkel.mtl.data import MultitaskDataLoader, MultitaskDataset
from snorkel.mtl.model import MultitaskModel
from snorkel.mtl.modules.utils import ce_loss, softmax
from snorkel.mtl.scorer import Scorer
from snorkel.mtl.task import Task
from snorkel.mtl.trainer import Trainer
from snorkel.slicing.utils import add_slice_labels, convert_to_slice_tasks
from snorkel.types import DataPoint

SEED = 123


# TODO: update me!
@slicing_function()
def f(x: DataPoint) -> int:
    return 1 if x.x1 > 0.75 else 0


@slicing_function()
def g(x: DataPoint) -> int:
    return 1 if x.x2 > 0.75 else 0


N_TRAIN = 1000
N_VALID = 100


class SlicingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_config = {"n_epochs": 2, "progress_bar": False}
        cls.logger_config = {"counter_unit": "epochs", "evaluation_freq": 0.25}    
    
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
        dataloaders[0] = add_slice_labels(task1, dataloaders[0], S_train, slice_names)
        dataloaders[1] = add_slice_labels(task1, dataloaders[1], S_valid, slice_names)

        self.assertEqual(len(dataloaders[0].task_to_label_dict), 8)
        self.assertIn("task1", dataloaders[0].task_to_label_dict)
        self.assertIn("task1_slice:f_ind", dataloaders[0].task_to_label_dict)
        self.assertIn("task1_slice:f_pred", dataloaders[0].task_to_label_dict)
        self.assertIn("task1_slice:g_ind", dataloaders[0].task_to_label_dict)
        self.assertIn("task1_slice:g_pred", dataloaders[0].task_to_label_dict)
        self.assertIn("task1_slice:base_ind", dataloaders[0].task_to_label_dict)
        self.assertIn("task1_slice:base_pred", dataloaders[0].task_to_label_dict)
        self.assertIn("task2", dataloaders[0].task_to_label_dict)

        # Convert to slice tasks
        task1_tasks = convert_to_slice_tasks(task1, slice_names)
        tasks = task1_tasks + [task2]
        model = MultitaskModel(tasks=tasks)
        
        # Train
        trainer = Trainer(**self.trainer_config, **self.logger_config)
        trainer.train_model(model, dataloaders)
        scores = model.score(dataloaders)

        self.assertGreater(scores["task1/TestData/train/accuracy"], 0.9)
        self.assertGreater(scores["task1/TestData/valid/accuracy"], 0.9)
        self.assertGreater(scores["task1_slice:f_ind/TestData/valid/accuracy"], 0.9)
        self.assertGreater(scores["task1_slice:f_pred/TestData/valid/accuracy"], 0.9)
        self.assertGreater(scores["task1_slice:g_ind/TestData/valid/accuracy"], 0.9)
        self.assertGreater(scores["task1_slice:g_pred/TestData/valid/accuracy"], 0.9)
        self.assertGreater(scores["task1_slice:base_ind/TestData/valid/accuracy"], 0.9)
        self.assertGreater(scores["task1_slice:base_pred/TestData/valid/accuracy"], 0.9)
        self.assertGreater(scores["task2/TestData/valid/accuracy"], 0.9)
        self.assertGreater(scores["task2/TestData/valid/accuracy"], 0.9)
        
        scores = model.score(dataloaders)
        print(scores)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_data(n):

    X = np.random.random((n, 2)) * 2 - 1
    Y = np.zeros((n, 2))
    Y[:, 0] = (X[:, 0] > X[:, 1] + 0.5).astype(int) + 1
    Y[:, 1] = (X[:, 0] > X[:, 1] + 0.25).astype(int) + 1
    
    df = pd.DataFrame(
        {"x1": X[:,0],
         "x2": X[:,1],
         "y1": Y[:,0],
         "y2": Y[:,1],
        }
    )
    return df


def create_dataloader(df, split):
    Y_dict = {}
    task_to_label_dict = {}
    
    Y_dict[f"task1_labels"] = torch.LongTensor(df["y1"])
    task_to_label_dict["task1"] = "task1_labels"
    
    Y_dict[f"task2_labels"] = torch.LongTensor(df["y2"])
    task_to_label_dict["task2"] = "task2_labels"

    dataset = MultitaskDataset(
        name="TestData", 
        X_dict={"coordinates": 
                torch.stack((torch.Tensor(df["x1"]), torch.Tensor(df["x2"])), dim=1)}, 
        Y_dict=Y_dict
    )

    dataloader = MultitaskDataLoader(
        task_to_label_dict=task_to_label_dict,
        dataset=dataset,
        split=split,
        batch_size=4,
        shuffle=(split == "train"),
    )
    return dataloader


def create_task(task_name, module_suffixes):
    module_pool = nn.ModuleDict(
        {
            f"linear1{module_suffixes[0]}": nn.Linear(2, 10),
            f"linear2{module_suffixes[1]}": nn.Linear(10, 2),
        }
    )

    task_flow = [
        {
            "name": "first_layer",
            "module": f"linear1{module_suffixes[0]}",
            "inputs": [("_input_", "coordinates")],
        },
        {
            "name": "second_layer",
            "module": f"linear2{module_suffixes[1]}",
            "inputs": [("first_layer", 0)],
        },
    ]

    task = Task(
        name=task_name,
        module_pool=module_pool,
        task_flow=task_flow,
        loss_func=partial(ce_loss, "second_layer"),
        output_func=partial(softmax, "second_layer"),
        scorer=Scorer(metrics=["accuracy"]),
    )

    return task