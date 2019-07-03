import unittest

import torch

from snorkel.analysis.utils import set_seed
from snorkel.end_model.batch_schedulers import SequentialScheduler, ShuffledScheduler
from snorkel.end_model.data import MultitaskDataLoader, MultitaskDataset

dataset1 = MultitaskDataset(
    "d1",
    "train",
    X_dict={"data": [0, 1, 2, 3, 4]},
    Y_dict={"labels": torch.LongTensor([1, 1, 1, 1, 1])},
)
dataset2 = MultitaskDataset(
    "d2",
    "train",
    X_dict={"data": [5, 6, 7, 8, 9]},
    Y_dict={"labels": torch.LongTensor([2, 2, 2, 2, 2])},
)

dataloader1 = MultitaskDataLoader(dataset1, batch_size=2)
dataloader2 = MultitaskDataLoader(dataset2, batch_size=2)
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
