import unittest

import torch

from snorkel.classification.data import DictDataLoader, DictDataset
from snorkel.classification.training.schedulers import (
    SequentialScheduler,
    ShuffledScheduler,
)
from snorkel.utils import set_seed

dataset1 = DictDataset(
    "d1",
    "train",
    X_dict={"data": [0, 1, 2, 3, 4]},
    Y_dict={"labels": torch.LongTensor([1, 1, 1, 1, 1])},
)
dataset2 = DictDataset(
    "d2",
    "train",
    X_dict={"data": [5, 6, 7, 8, 9]},
    Y_dict={"labels": torch.LongTensor([2, 2, 2, 2, 2])},
)

dataloader1 = DictDataLoader(dataset1, batch_size=2)
dataloader2 = DictDataLoader(dataset2, batch_size=2)
dataloaders = [dataloader1, dataloader2]


class SequentialTest(unittest.TestCase):
    def test_sequential(self):
        scheduler = SequentialScheduler()
        data = []
        for (batch, dl) in scheduler.get_batches(dataloaders):
            X_dict, Y_dict = batch
            data.extend(X_dict["data"])
        self.assertEqual(data, sorted(data))

    def test_shuffled(self):
        set_seed(123)
        scheduler = ShuffledScheduler()
        data = []
        for (batch, dl) in scheduler.get_batches(dataloaders):
            X_dict, Y_dict = batch
            data.extend(X_dict["data"])
        self.assertNotEqual(data, sorted(data))


if __name__ == "__main__":
    unittest.main()
