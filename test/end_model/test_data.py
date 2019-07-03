import unittest

import torch

from snorkel.end_model.data import MultitaskDataLoader, MultitaskDataset


class DatasetTest(unittest.TestCase):
    def test_mtl_dataset(self):
        """Unit test of MultitaskDataset"""

        x1 = [
            torch.Tensor([1]),
            torch.Tensor([1, 2]),
            torch.Tensor([1, 2, 3]),
            torch.Tensor([1, 2, 3, 4]),
            torch.Tensor([1, 2, 3, 4, 5]),
        ]

        y1 = torch.Tensor([0, 0, 0, 0, 0])

        dataset = MultitaskDataset(
            X_dict={"data1": x1}, Y_dict={"label1": y1}, name="new_data", split="train"
        )

        # Check if the dataset is correctly constructed
        self.assertTrue(torch.equal(dataset[0][0]["data1"], x1[0]))
        self.assertTrue(torch.equal(dataset[0][1]["label1"], y1[0]))

    def test_mtl_dataloader(self):
        """Unit test of MultitaskDataloader"""

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

        dataset = MultitaskDataset(
            name="new_data",
            split="train",
            X_dict={"data1": x1, "data2": x2},
            Y_dict={"label1": y1, "label2": y2},
        )

        dataloader1 = MultitaskDataLoader(
            task_to_label_dict={"task1": "label1"}, dataset=dataset, batch_size=2
        )

        x_batch, y_batch = next(iter(dataloader1))

        # Check if the dataloader is correctly constructed
        self.assertEqual(dataloader1.task_to_label_dict, {"task1": "label1"})
        self.assertEqual(dataloader1.dataset.split, "train")
        self.assertTrue(torch.equal(x_batch["data1"], torch.Tensor([[1, 0], [1, 2]])))
        self.assertTrue(
            torch.equal(
                x_batch["data2"], torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0]])
            )
        )
        self.assertTrue(torch.equal(y_batch["label1"], torch.Tensor([0, 0])))
        self.assertTrue(torch.equal(y_batch["label2"], torch.Tensor([1, 1])))

        dataloader2 = MultitaskDataLoader(
            task_to_label_dict={"task2": "label2"}, dataset=dataset, batch_size=3
        )

        x_batch, y_batch = next(iter(dataloader2))

        # Check if the dataloader with differet batch size is correctly constructed
        self.assertEqual(dataloader2.task_to_label_dict, {"task2": "label2"})
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
        self.assertTrue(torch.equal(y_batch["label1"], torch.Tensor([0, 0, 0])))
        self.assertTrue(torch.equal(y_batch["label2"], torch.Tensor([1, 1, 1])))

        y3 = [
            torch.Tensor([2]),
            torch.Tensor([2]),
            torch.Tensor([2]),
            torch.Tensor([2]),
            torch.Tensor([2]),
        ]

        dataset.Y_dict["label2"] = y3

        x_batch, y_batch = next(iter(dataloader1))
        # Check dataloader is correctly updated with update dataset
        self.assertTrue(
            torch.equal(
                x_batch["data2"], torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0]])
            )
        )
        self.assertTrue(torch.equal(y_batch["label2"], torch.Tensor([[2], [2]])))

        x_batch, y_batch = next(iter(dataloader2))
        self.assertTrue(
            torch.equal(
                x_batch["data2"],
                torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0], [1, 2, 3, 0, 0]]),
            )
        )
        self.assertTrue(torch.equal(y_batch["label2"], torch.Tensor([[2], [2], [2]])))


if __name__ == "__main__":
    unittest.main()
