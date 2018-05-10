from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from numpy.linalg import norm
from snorkel.learning.pytorch import LSTM
from pytorch_test_base import PyTorchTestBase
import unittest


class TestModelReloading(PyTorchTestBase):

    def test_lstm_reloading(self):
        train_kwargs = {
            'lr':            0.01,
            'embedding_dim': 50,
            'hidden_dim':    50,
            'n_epochs':      2,
            'dropout':       0.25
        }

        train_kwargs['num_layers'] = 1
        train_kwargs['bidirectional'] = False
        lstm1 = LSTM()
        lstm1.train(self.train_cands, self.train_marginals, **train_kwargs)
        marginals1_before = lstm1.marginals(self.test_cands)

        train_kwargs['num_layers'] = 2
        train_kwargs['bidirectional'] = True
        lstm2 = LSTM()
        lstm2.train(self.train_cands, self.train_marginals, **train_kwargs)
        marginals2_before = lstm2.marginals(self.test_cands)

        lstm1.save('lstm1')
        lstm2.save('lstm2')

        lstm1 = LSTM()
        lstm1.load('lstm1')
        marginals1_after = lstm1.marginals(self.test_cands)
        self.assertTrue(norm(marginals1_before - marginals1_after) < 1e-5)

        lstm2 = LSTM()
        lstm2.load('lstm2')
        marginals2_after = lstm2.marginals(self.test_cands)
        self.assertTrue(norm(marginals2_before - marginals2_after) < 1e-5)

if __name__ == '__main__':
    unittest.main()
