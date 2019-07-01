import unittest
from collections import Counter

import numpy as np
import scipy.sparse as sparse
import torch

from snorkel.model.utils import pred_to_prob, rargmax, recursive_merge_dicts, split_data


class UtilsTest(unittest.TestCase):
    def test_rargmax(self):
        x = np.array([2, 1, 2])
        np.random.seed(1)
        self.assertEqual(sorted(list(set(rargmax(x) for _ in range(10)))), [0, 2])

    def test_pred_to_prob(self):
        x = torch.tensor([1, 2, 2, 1])
        target = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]])
        self.assertTrue(
            (pred_to_prob(x, 2).float() == target.float()).sum()
            == torch.prod(torch.tensor(target.shape))
        )

    def test_recursive_merge_dicts(self):
        x = {"foo": {"Foo": {"FOO": 1}}, "bar": 2, "baz": 3}
        y = {"FOO": 4, "bar": 5}
        z = {"foo": 6}
        w = recursive_merge_dicts(x, y, verbose=False)
        self.assertEqual(w["bar"], 5)
        self.assertEqual(w["foo"]["Foo"]["FOO"], 4)
        with self.assertRaises(ValueError):
            recursive_merge_dicts(x, z, verbose=False)

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
