import unittest

import torch

from snorkel.classification.utils import (
    collect_flow_outputs_by_suffix,
    list_to_tensor,
    pad_batch,
)


class UtilsTest(unittest.TestCase):
    def test_pad_batch(self):
        batch = [torch.Tensor([1, 2]), torch.Tensor([3]), torch.Tensor([4, 5, 6])]
        padded_batch, mask_batch = pad_batch(batch)

        self.assertTrue(
            torch.equal(padded_batch, torch.Tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]]))
        )
        self.assertTrue(
            torch.equal(mask_batch, torch.Tensor([[0, 0, 1], [0, 1, 1], [0, 0, 0]]))
        )

        padded_batch, mask_batch = pad_batch(batch, max_len=2)

        self.assertTrue(
            torch.equal(padded_batch, torch.Tensor([[1, 2], [3, 0], [4, 5]]))
        )
        self.assertTrue(torch.equal(mask_batch, torch.Tensor([[0, 0], [0, 1], [0, 0]])))

        padded_batch, mask_batch = pad_batch(batch, pad_value=-1)

        self.assertTrue(
            torch.equal(
                padded_batch, torch.Tensor([[1, 2, -1], [3, -1, -1], [4, 5, 6]])
            )
        )
        self.assertTrue(
            torch.equal(mask_batch, torch.Tensor([[0, 0, 1], [0, 1, 1], [0, 0, 0]]))
        )

        padded_batch, mask_batch = pad_batch(batch, left_padded=True)

        self.assertTrue(
            torch.equal(padded_batch, torch.Tensor([[0, 1, 2], [0, 0, 3], [4, 5, 6]]))
        )
        self.assertTrue(
            torch.equal(mask_batch, torch.Tensor([[1, 0, 0], [1, 1, 0], [0, 0, 0]]))
        )

        padded_batch, mask_batch = pad_batch(batch, max_len=2, left_padded=True)

        self.assertTrue(
            torch.equal(padded_batch, torch.Tensor([[1, 2], [0, 3], [5, 6]]))
        )
        self.assertTrue(torch.equal(mask_batch, torch.Tensor([[0, 0], [1, 0], [0, 0]])))

    def test_list_to_tensor(self):
        # list of 1-D tensor with the different length
        batch = [torch.Tensor([1, 2]), torch.Tensor([3]), torch.Tensor([4, 5, 6])]

        padded_batch = list_to_tensor(batch)

        self.assertTrue(
            torch.equal(padded_batch, torch.Tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]]))
        )

        # list of 1-D tensor with the same length
        batch = [
            torch.Tensor([1, 2, 3]),
            torch.Tensor([4, 5, 6]),
            torch.Tensor([7, 8, 9]),
        ]

        padded_batch = list_to_tensor(batch)

        self.assertTrue(
            torch.equal(padded_batch, torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        )

        # list of 2-D tensor with the same size
        batch = [
            torch.Tensor([[1, 2, 3], [1, 2, 3]]),
            torch.Tensor([[4, 5, 6], [4, 5, 6]]),
            torch.Tensor([[7, 8, 9], [7, 8, 9]]),
        ]

        padded_batch = list_to_tensor(batch)

        self.assertTrue(
            torch.equal(
                padded_batch,
                torch.Tensor(
                    [
                        [[1, 2, 3], [1, 2, 3]],
                        [[4, 5, 6], [4, 5, 6]],
                        [[7, 8, 9], [7, 8, 9]],
                    ]
                ),
            )
        )

        # list of tensor with the different size
        batch = [
            torch.Tensor([[1, 2], [2, 3]]),
            torch.Tensor([4, 5, 6]),
            torch.Tensor([7, 8, 9, 0]),
        ]

        padded_batch = list_to_tensor(batch)

        self.assertTrue(
            torch.equal(
                padded_batch, torch.Tensor([[1, 2, 2, 3], [4, 5, 6, 0], [7, 8, 9, 0]])
            )
        )

    def test_collect_flow_outputs_by_suffix(self):
        flow_dict = {
            "a_pred_head": torch.Tensor([1]),
            "b_pred_head": torch.Tensor([2]),
            "c_pred": torch.Tensor([3]),
        }
        outputs = collect_flow_outputs_by_suffix(flow_dict, "_head")
        self.assertIn(torch.Tensor([1]), outputs)
        self.assertIn(torch.Tensor([2]), outputs)

    if __name__ == "__main__":
        unittest.main()
