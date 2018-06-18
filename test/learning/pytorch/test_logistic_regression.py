from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from snorkel.learning.pytorch import LogisticRegression
from pytorch_test_base import PyTorchTestBase
import unittest


class TestLogisticRegression(PyTorchTestBase):

    def test_lr(self):
        train_kwargs = {
            'lr':            0.01,
            'n_epochs':      10,
            'print_freq':    1,
        }

        lr = LogisticRegression()
        lr.train(self.F_train, self.train_marginals, **train_kwargs)
        _, _, f1 = lr.score(self.F_dev, self.L_gold_dev)
        self.assertTrue(f1 > .7)

    def test_lr_with_dev_set(self):
        train_kwargs = {
            'lr':            0.01,
            'n_epochs':      10,
            'print_freq':    1,
        }

        lr = LogisticRegression()
        lr.train(self.F_train, self.train_marginals, X_dev=self.F_dev, Y_dev=self.L_gold_dev, **train_kwargs)
        _, _, f1 = lr.score(self.F_dev, self.L_gold_dev)
        self.assertTrue(f1 > .7)


if __name__ == '__main__':
    unittest.main()
