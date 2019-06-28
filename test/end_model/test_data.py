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
            X_dict={"data1": x1}, Y_dict={"label1": y1}, name="new_data"
        )

        # Check if the dataset is correctly constructed
        self.assertTrue(torch.equal(dataset[0][0]["data1"], x1[0]))
        self.assertTrue(torch.equal(dataset[0][1]["label1"], y1[0]))

        x2 = [
            torch.Tensor([1, 2, 3, 4, 5]),
            torch.Tensor([1, 2, 3, 4]),
            torch.Tensor([1, 2, 3]),
            torch.Tensor([1, 2]),
            torch.Tensor([1]),
        ]

        dataset.add_features(X_dict={"data2": x2})

        # Check add one more feature to dataset
        self.assertTrue(torch.equal(dataset[0][0]["data2"], x2[0]))

        y2 = torch.Tensor([1, 1, 1, 1, 1])

        dataset.add_labels(Y_dict={"label2": y2})

        # Check add one more label to dataset
        self.assertTrue(torch.equal(dataset[0][1]["label2"], y2[0]))

        dataset.remove_label(label_name="label1")

        # Check remove one more label to dataset
        self.assertTrue("label1" not in dataset.Y_dict)

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
            X_dict={"data1": x1, "data2": x2},
            Y_dict={"label1": y1, "label2": y2},
            name="new_data",
        )

        dataloader1 = MultitaskDataLoader(
            task_to_label_dict={"task1": "label1"},
            dataset=dataset,
            split="train",
            batch_size=2,
        )

        x_batch, y_batch = next(iter(dataloader1))

        # Check if the dataloader is correctly constructed
        self.assertEqual(dataloader1.task_to_label_dict, {"task1": "label1"})
        self.assertEqual(dataloader1.split, "train")
        self.assertTrue(torch.equal(x_batch["data1"], torch.Tensor([[1, 0], [1, 2]])))
        self.assertTrue(
            torch.equal(
                x_batch["data2"], torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0]])
            )
        )
        self.assertTrue(torch.equal(y_batch["label1"], torch.Tensor([0, 0])))
        self.assertTrue(torch.equal(y_batch["label2"], torch.Tensor([1, 1])))

        dataloader2 = MultitaskDataLoader(
            task_to_label_dict={"task2": "label2"},
            dataset=dataset,
            split="test",
            batch_size=3,
        )

        x_batch, y_batch = next(iter(dataloader2))

        # Check if the dataloader with differet batch size is correctly constructed
        self.assertEqual(dataloader2.task_to_label_dict, {"task2": "label2"})
        self.assertEqual(dataloader2.split, "test")
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
