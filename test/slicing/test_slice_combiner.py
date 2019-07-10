import unittest

import torch

from snorkel.analysis.utils import set_seed
from snorkel.slicing.modules.slice_combiner import SliceCombinerModule


class SliceCombinerTest(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        set_seed(123)

    def test_init_and_forward(self):
        batch_size = 4
        h_dim = 20

        outputs = {
            "_input_": {"data": torch.FloatTensor(batch_size, 2).uniform_(0, 1)},
            "linear": [torch.FloatTensor(batch_size, 2).uniform_(0, 1)],
            "task_slice:base_ind_head": [
                torch.FloatTensor(batch_size, 2).uniform_(0, 1)
            ],
            "task_slice:base_pred_transform": [
                torch.FloatTensor(batch_size, 20).uniform_(0, 1)
            ],
            "task_slice:base_pred_head": [
                torch.FloatTensor(batch_size, 2).uniform_(0, 1)
            ],
        }
        combiner_module = SliceCombinerModule()
        combined_rep = combiner_module(outputs)
        self.assertEquals(tuple(combined_rep.shape), (batch_size, h_dim))
