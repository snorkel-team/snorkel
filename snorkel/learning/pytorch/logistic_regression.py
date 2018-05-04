from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn as nn
from builtins import *
from scipy.sparse import issparse

from snorkel.learning.pytorch.disc_learning import TorchNoiseAwareModel

class LogisticRegression(TorchNoiseAwareModel):
    representation = False
    
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        out = self.linear(X)
        return self.sigmoid(out)
        
    def _check_input(self, X):
        if issparse(X):
            raise Exception("Sparse input matrix. Use SparseLogisticRegression")
        return X

    def marginals_batch(self, X_test):
        X_test = self._check_input(X_test)
        output = self.forward(X_test)
        return self.sigmoid(output)
