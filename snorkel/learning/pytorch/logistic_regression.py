from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import numpy as np
from scipy.sparse import issparse
import torch
import torch.nn as nn
import warnings

from .noise_aware_model import TorchNoiseAwareModel


class LogisticRegression(TorchNoiseAwareModel):
    representation = False

    def train(self, X_train, Y_train, X_dev=None, **kwargs):
        # Preprocesses data
        X_train = self._preprocess_data(X_train)
        if X_dev is not None:
            X_dev = self._preprocess_data(X_dev)

        # Intercepts kwargs to set input_dim
        if "input_dim" in kwargs:
            warnings.warn("Overwriting train kwarg 'input_dim'")
        kwargs["input_dim"] = X_train.shape[1]

        super(LogisticRegression, self).train(X_train, Y_train, X_dev=X_dev, **kwargs)

    def _build_model(self, input_dim=None, **model_kwargs):
        if input_dim is None:
            raise ValueError("Kwarg input_dim cannot be None.")

        self.linear = nn.Linear(input_dim, self.cardinality if self.cardinality > 2 else 1)

    def _pytorch_outputs(self, X, batch_size):
        # TODO: This code is mostly duplicated in .rnn.rnn_base.RNNBase::_pytorch_outputs()
        X = self._preprocess_data(X)

        if not batch_size:
            batch_size = len(X)
        outputs = torch.Tensor([])

        for batch in range(0, len(X), batch_size):
            if batch_size > len(X[batch:batch+batch_size]):
                batch_size = len(X[batch:batch+batch_size])
            output = self.forward(X[batch:batch+batch_size])

            # TODO: Does skipping the cat when there is only one batch speed things up significantly?
            if self.cardinality == 2:
                outputs = torch.cat((outputs, output.view(-1)), 0)
            else:
                outputs = torch.cat((outputs, output), 0)
        return outputs

    def forward(self, X):
        return self.linear(X)
        
    def _preprocess_data(self, X):
        if issparse(X):
            warnings.warn("Converting sparse matrix to dense.")
            X = X.todense()

        if isinstance(X, np.ndarray):
            if X.dtype == np.float32:
                X = torch.from_numpy(X)
            else:
                warnings.warn("Casting input matrix to float32.")
                X = torch.tensor(X, dtype=torch.float32)

        return X
