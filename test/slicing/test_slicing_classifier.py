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
        data_points = [SimpleNamespace(num=num) for num in DATA]
        applier = SFApplier([f, g])
        cls.S = applier.apply(data_points, progress_bar=False)

    def test_classifier(self):
        hidden_dim = 10
        input_dim = 2

        representation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        input_data_key = "test_data"
        task_name = "test_task"
        slice_names = ["hello", "world"]
        slicing_cls = BinarySlicingClassifier(
            representation_net=representation_net,
            head_dim=hidden_dim,
            slice_names=slice_names,
            input_data_key=input_data_key,
            task_name=task_name,
            scorer=Scorer(metrics=["f1"]),
        )

        # Repeated data value for [N x 2] dim Tensor
        X = torch.FloatTensor([(x, x) for x in DATA])
        # Alternating labels
        Y = torch.LongTensor([int(i % 2 == 0) for i in range(len(DATA))])

        dataset_name = "test_dataset"

        splits = ["train", "valid"]
        datasets = [
            create_dataset(X, Y, split, dataset_name, input_data_key, task_name)
            for split in splits
        ]

        dataloaders = [
            slicing_cls.make_slice_dataloader(dataset=ds, S=self.S, batch_size=4)
            for ds in datasets
        ]

        trainer = Trainer(n_epochs=1)
        trainer.fit(slicing_cls, dataloaders)

        results = slicing_cls.score_slices(dataloaders)

        # Check that we eval on 'pred' labels
        self.assertIn(f"{task_name}/test_dataset/train/f1", results)
        self.assertIn(f"{task_name}/test_dataset/valid/f1", results)
        self.assertIn(f"{task_name}_slice:hello_pred/test_dataset/train/f1", results)
        self.assertIn(f"{task_name}_slice:hello_pred/test_dataset/valid/f1", results)
        self.assertIn(f"{task_name}_slice:world_pred/test_dataset/train/f1", results)
        self.assertIn(f"{task_name}_slice:world_pred/test_dataset/valid/f1", results)

        # No 'ind' labels!
        self.assertNotIn(f"{task_name}_slice:hello_ind/test_dataset/train/f1", results)
        self.assertNotIn(f"{task_name}_slice:hello_ind/test_dataset/valid/f1", results)
        self.assertNotIn(f"{task_name}_slice:world_ind/test_dataset/train/f1", results)
        self.assertNotIn(f"{task_name}_slice:world_ind/test_dataset/valid/f1", results)


def create_dataset(X, Y, split, dataset_name, input_name, task_name):
    return DictDataset(
        name=dataset_name, split=split, X_dict={input_name: X}, Y_dict={task_name: Y}
    )
