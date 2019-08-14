import unittest

import torch

from snorkel.classification import DictDataLoader, DictDataset
from snorkel.classification.data import (
    DEFAULT_DATASET_NAME,
    DEFAULT_INPUT_DATA_KEY,
    DEFAULT_TASK_NAME,
)


class DatasetTest(unittest.TestCase):
    def test_classifier_dataset(self):
        """Unit test of DictDataset"""

        x1 = [
            torch.Tensor([1]),
            torch.Tensor([1, 2]),
            torch.Tensor([1, 2, 3]),
            torch.Tensor([1, 2, 3, 4]),
            torch.Tensor([1, 2, 3, 4, 5]),
        ]

        y1 = torch.Tensor([0, 0, 0, 0, 0])

        dataset = DictDataset(
            X_dict={"data1": x1}, Y_dict={"task1": y1}, name="new_data", split="train"
        )

        # Check if the dataset is correctly constructed
        self.assertTrue(torch.equal(dataset[0][0]["data1"], x1[0]))
        self.assertTrue(torch.equal(dataset[0][1]["task1"], y1[0]))
        self.assertEqual(
            repr(dataset),
            "DictDataset(name=new_data, X_keys=['data1'], Y_keys=['task1'])",
        )

        # Test from_tensors inits with default values
        dataset = DictDataset.from_tensors(x1, y1, "train")
        self.assertEqual(
            repr(dataset),
            f"DictDataset(name={DEFAULT_DATASET_NAME}, "
            f"X_keys=['{DEFAULT_INPUT_DATA_KEY}'], Y_keys=['{DEFAULT_TASK_NAME}'])",
        )

    def test_classifier_dataloader(self):
        """Unit test of DictDataLoader"""

        x1 = [
            torch.Tensor([1]),
            torch.Tensor([1, 2]),
            torch.Tensor([1, 2, 3]),
            torch.Tensor([1, 2, 3, 4]),
            torch.Tensor([1, 2, 3, 4, 5]),
        ]

        y1 = torch.Tensor([0, 0, 0, 0, 0])

        x2 = [
            torch.Tensor([1, 2, 3, 4, 5]),
            torch.Tensor([1, 2, 3, 4]),
            torch.Tensor([1, 2, 3]),
            torch.Tensor([1, 2]),
            torch.Tensor([1]),
        ]

        y2 = torch.Tensor([1, 1, 1, 1, 1])

        dataset = DictDataset(
            name="new_data",
            split="train",
            X_dict={"data1": x1, "data2": x2},
            Y_dict={"task1": y1, "task2": y2},
        )

        dataloader1 = DictDataLoader(dataset=dataset, batch_size=2)

        x_batch, y_batch = next(iter(dataloader1))

        # Check if the dataloader is correctly constructed
        self.assertEqual(dataloader1.dataset.split, "train")
        self.assertTrue(torch.equal(x_batch["data1"], torch.Tensor([[1, 0], [1, 2]])))
        self.assertTrue(
            torch.equal(
                x_batch["data2"], torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0]])
            )
        )
        self.assertTrue(torch.equal(y_batch["task1"], torch.Tensor([0, 0])))
        self.assertTrue(torch.equal(y_batch["task2"], torch.Tensor([1, 1])))

        dataloader2 = DictDataLoader(dataset=dataset, batch_size=3)

        x_batch, y_batch = next(iter(dataloader2))

        # Check if the dataloader with differet batch size is correctly constructed
        self.assertEqual(dataloader2.dataset.split, "train")
        self.assertTrue(
            torch.equal(
                x_batch["data1"], torch.Tensor([[1, 0, 0], [1, 2, 0], [1, 2, 3]])
            )
        )
        self.assertTrue(
            torch.equal(
                x_batch["data2"],
                torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0], [1, 2, 3, 0, 0]]),
            )
        )
        self.assertTrue(torch.equal(y_batch["task1"], torch.Tensor([0, 0, 0])))
        self.assertTrue(torch.equal(y_batch["task2"], torch.Tensor([1, 1, 1])))

        y3 = [
            torch.Tensor([2]),
            torch.Tensor([2]),
            torch.Tensor([2]),
            torch.Tensor([2]),
            torch.Tensor([2]),
        ]

        dataset.Y_dict["task2"] = y3

        x_batch, y_batch = next(iter(dataloader1))
        # Check dataloader is correctly updated with update dataset
        self.assertTrue(
            torch.equal(
                x_batch["data2"], torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0]])
            )
        )
        self.assertTrue(torch.equal(y_batch["task2"], torch.Tensor([[2], [2]])))

        x_batch, y_batch = next(iter(dataloader2))
        self.assertTrue(
            torch.equal(
                x_batch["data2"],
                torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0], [1, 2, 3, 0, 0]]),
            )
        )
        self.assertTrue(torch.equal(y_batch["task2"], torch.Tensor([[2], [2], [2]])))


if __name__ == "__main__":
    unittest.main()
