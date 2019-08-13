import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from snorkel.analysis import Scorer
from snorkel.classification import DictDataset, Trainer
from snorkel.slicing import BinarySlicingClassifier, SFApplier, slicing_function


@slicing_function()
def f(x) -> int:
    return x.num > 42


@slicing_function()
def g(x) -> int:
    return x.num > 10


DATA = [3, 43, 12, 9, 3]


class SliceCombinerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define S_matrix
        data_points = [SimpleNamespace(num=num) for num in DATA]
        applier = SFApplier([f, g])
        cls.S = applier.apply(data_points, progress_bar=False)

        # Define base architecture
        cls.hidden_dim = 10
        cls.mlp = nn.Sequential(
            nn.Linear(2, cls.hidden_dim),
            nn.Linear(cls.hidden_dim, cls.hidden_dim),
            nn.ReLU(),
        )

        # Define model parameters
        cls.data_name = "test_data"
        cls.task_name = "test_task"

        # Define slice metadata
        cls.slice_names = ["hello", "world"]

        # Define datasets
        # Repeated data value for [N x 2] dim Tensor
        X = torch.FloatTensor([(x, x) for x in DATA])
        # Alternating labels
        Y = torch.LongTensor([int(i % 2 == 0) for i in range(len(DATA))])

        dataset_name = "test_dataset"
        splits = ["train", "valid"]
        cls.datasets = [
            create_dataset(X, Y, split, dataset_name, cls.data_name, cls.task_name)
            for split in splits
        ]

    def test_scores(self):
        slice_model = BinarySlicingClassifier(
            base_architecture=self.mlp,
            head_dim=self.hidden_dim,
            slice_names=self.slice_names,
            input_data_key=self.data_name,
            task_name=self.task_name,
            scorer=Scorer(metrics=["f1"]),
        )

        # Make dataloaders
        dataloaders = [
            slice_model.make_slice_dataloader(dataset=ds, S=self.S, batch_size=4)
            for ds in self.datasets
        ]
        train_dl, valid_dl = tuple(dataloaders)

        # Train model
        trainer = Trainer(n_epochs=1)
        trainer.fit(slice_model, [train_dl])

        # Eval overall
        scores = slice_model.score([valid_dl])
        # All labels should appears in .score() output
        self.assertIn("test_task/test_dataset/valid/f1", scores)
        self.assertIn("test_task_slice:hello_pred/test_dataset/valid/f1", scores)
        self.assertIn("test_task_slice:world_pred/test_dataset/valid/f1", scores)
        self.assertIn("test_task_slice:hello_ind/test_dataset/valid/f1", scores)
        self.assertIn("test_task_slice:world_ind/test_dataset/valid/f1", scores)

        # Eval on slices
        slice_scores = slice_model.score_slices([valid_dl])
        # Check that we eval on 'pred' labels in .score_slices() output
        self.assertIn("test_task/test_dataset/valid/f1", slice_scores)
        self.assertIn("test_task_slice:hello_pred/test_dataset/valid/f1", slice_scores)
        self.assertIn("test_task_slice:world_pred/test_dataset/valid/f1", slice_scores)

        # No 'ind' labels!
        self.assertNotIn(
            "test_task_slice:hello_ind/test_dataset/valid/f1", slice_scores
        )
        self.assertNotIn(
            "test_task_slice:world_ind/test_dataset/valid/f1", slice_scores
        )


def create_dataset(X, Y, split, dataset_name, input_name, task_name):
    return DictDataset(
        name=dataset_name, split=split, X_dict={input_name: X}, Y_dict={task_name: Y}
    )
