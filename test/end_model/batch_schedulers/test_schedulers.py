import unittest

import torch

from snorkel.analysis.utils import set_seed
from snorkel.end_model.batch_schedulers import SequentialScheduler, ShuffledScheduler
from snorkel.end_model.data import SnorkelDataLoader, SnorkelDataset

dataset1 = SnorkelDataset(
    "d1", "train", {"data": ["a", "b", "c"]}, {"labels": torch.LongTensor([1, 2, 3])}
)
dataset2 = SnorkelDataset(
    "d2", "train", {"data": ["d", "e", "f"]}, {"labels": torch.LongTensor([4, 5, 6])}
)

dataloader1 = SnorkelDataLoader(dataset1, batch_size=2)
dataloader2 = SnorkelDataLoader(dataset2, batch_size=2)
dataloaders = [dataloader1, dataloader2]


class SequentialTest(unittest.TestCase):
    def test_sequential(self):
        scheduler = SequentialScheduler()
        data = []
        for (batch, dl) in scheduler.get_batches(dataloaders):
            X_dict, Y_dict = batch
            data.extend(X_dict["data"])
        self.assertEqual(data, ["a", "b", "c", "d", "e", "f"])

    def test_shuffled(self):
        set_seed(123)
        scheduler = ShuffledScheduler()
        data = []
        for (batch, dl) in scheduler.get_batches(dataloaders):
            X_dict, Y_dict = batch
            data.extend(X_dict["data"])
        self.assertNotEqual(data, ["a", "b", "c", "d", "e", "f"])


if __name__ == "__main__":
    unittest.main()
