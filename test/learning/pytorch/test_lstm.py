from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import math
from snorkel.learning.pytorch import LSTM
from pytorch_test_base import PyTorchTestBase
import unittest


class TestLSTM(PyTorchTestBase):

    def test_lstm_architectures(self):
        for num_layers in (1, 2):
            for bidirectional in (True, False):
                train_kwargs = {
                    'lr':            0.01,
                    'embedding_dim': 50,
                    'hidden_dim':    50,
                    'n_epochs':      2,
                    'dropout':       0.25,
                    'num_layers':    num_layers,
                    'bidirectional': bidirectional
                }

                lstm = LSTM()
                lstm.train(self.train_cands, self.train_marginals, **train_kwargs)
                _, _, f1 = lstm.score(self.test_cands, self.L_gold_test)
                self.assertFalse(math.isnan(f1))

    def test_lstm_with_dev_set(self):
        train_kwargs = {
            'lr':            0.01,
            'embedding_dim': 50,
            'hidden_dim':    50,
            'n_epochs':      2,
            'dropout':       0.25,
            'num_layers':    1,
            'bidirectional': False
        }

        lstm = LSTM()
        lstm.train(self.train_cands, self.train_marginals, X_dev=self.dev_cands, Y_dev=self.L_gold_dev, **train_kwargs)
        _, _, f1 = lstm.score(self.test_cands, self.L_gold_test)
        self.assertFalse(math.isnan(f1))


if __name__ == '__main__':
    unittest.main()
