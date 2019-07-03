import unittest

import numpy as np
import torch

from snorkel.classification.data import ClassifierDataLoader, ClassifierDataset, split_data


class DatasetTest(unittest.TestCase):
    def test_mtl_dataset(self):
        """Unit test of ClassifierDataset"""

        x1 = [
            torch.Tensor([1]),
            torch.Tensor([1, 2]),
            torch.Tensor([1, 2, 3]),
            torch.Tensor([1, 2, 3, 4]),
            torch.Tensor([1, 2, 3, 4, 5]),
        ]

        y1 = torch.Tensor([0, 0, 0, 0, 0])

        dataset = ClassifierDataset(
            X_dict={"data1": x1}, Y_dict={"label1": y1}, name="new_data", split="train"
        )

        # Check if the dataset is correctly constructed
        self.assertTrue(torch.equal(dataset[0][0]["data1"], x1[0]))
        self.assertTrue(torch.equal(dataset[0][1]["label1"], y1[0]))

    def test_mtl_dataloader(self):
        """Unit test of ClassifierDataLoader"""

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

        dataset = ClassifierDataset(
            name="new_data",
            split="train",
            X_dict={"data1": x1, "data2": x2},
            Y_dict={"label1": y1, "label2": y2},
        )

        dataloader1 = ClassifierDataLoader(
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

        dataloader2 = ClassifierDataLoader(
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

    def test_split_data(self):
        N = 1000
        K = 4

        X = np.arange(0, N)
        Y = np.random.randint(0, K, size=N).astype(int)

        # Creates splits of correct size
        splits = [800, 100, 100]
        Xs_1 = split_data(X, splits=splits, shuffle=False)
        for i, count in enumerate(splits):
            self.assertEqual(len(Xs_1[i]), count)

        # Accepts floats or ints
        splits = [0.8, 0.1, 0.1]
        Xs_2 = split_data(X, splits=splits, shuffle=False)
        for split in range(len(splits)):
            self.assertTrue(np.array_equal(Xs_2[split], Xs_1[split]))

        # Shuffles correctly
        Xs_3 = split_data(X, splits=splits, shuffle=True, seed=123)
        self.assertNotEqual(Xs_3[0][0], Xs_2[0][0])

        # Indices only
        splits = [0.8, 0.1, 0.1]
        Ys = split_data(Y, splits=splits, shuffle=False, index_only=True)
        self.assertGreater(max(Ys[0]), K)

        # Handles multiple inputs
        Xs, Ys = split_data(X, Y, splits=splits, shuffle=True, seed=123)
        self.assertEqual(Ys[0][0], Y[Xs[0][0]])

        # Confirm statification (correct proportion of labels in each split)
        Ys = split_data(Y, splits=splits, stratify_by=Y, seed=123)
        counts = [Counter(Y) for Y in Ys]
        for y in np.unique(Y):
            ratio0 = counts[0][y] / len(Ys[0])
            ratio1 = counts[1][y] / len(Ys[1])
            ratio2 = counts[2][y] / len(Ys[2])
            self.assertLess(abs(ratio0 - ratio1), 0.05)
            self.assertLess(abs(ratio0 - ratio2), 0.05)

        # Handles scipy.sparse matrices
        Z = sparse.csr_matrix([[1, 0, 1, 2], [0, 3, 0, 3], [1, 2, 3, 4], [5, 4, 3, 2]])
        splits = [0.75, 0.25]
        Zs = split_data(Z, splits=splits, shuffle=True, seed=123)
        self.assertEqual(Zs[0].shape, (3, 4))

        # Handles torch.Tensors
        W = torch.Tensor([[1, 0, 1, 2], [0, 3, 0, 3], [1, 2, 3, 4], [5, 4, 3, 2]])
        splits = [0.75, 0.25]
        Ws = split_data(W, splits=splits, shuffle=True, seed=123)
        self.assertEqual(Ws[0].shape, (3, 4))



if __name__ == "__main__":
    unittest.main()
