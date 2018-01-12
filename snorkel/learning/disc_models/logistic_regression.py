from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import numpy as np
import tensorflow as tf

from snorkel.learning.disc_learning import TFNoiseAwareModel
from scipy.sparse import csr_matrix, issparse
from time import time
from six.moves.cPickle import dump, load
from snorkel.learning.utils import LabelBalancer, reshape_marginals

SD = 0.1


class LogisticRegression(TFNoiseAwareModel):
    representation = False

    def _build_model(self, d=None, **kwargs):
        """
        Build model, setting logits and marginals ops.
        
        :param d: Number of features
        """
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)

        # Define inputs
        self.X = tf.placeholder(tf.float32, (None, d))
        self.Y = tf.placeholder(tf.float32, (None, self.cardinality)) \
            if self.cardinality > 2 else tf.placeholder(tf.float32, (None,))

        # Define parameters and logits
        k = self.cardinality if self.cardinality > 2 else 1
        self.w = tf.Variable(tf.random_normal((d, k), stddev=SD, seed=s1))
        self.b = tf.Variable(tf.random_normal((k,), stddev=SD, seed=s2))

        if self.deterministic:
            # TODO: Implement for categorical as well...
            if self.cardinality > 2:
                raise NotImplementedError(
                    "Deterministic mode not implemented for categoricals.")

            # Make deterministic
            # See: https://www.twosigma.com/insights/a-workaround-for-non-determinism-in-tensorflow
            f_w = tf.matmul(self.X, self.w)
            f_w_temp = tf.concat([f_w, tf.ones_like(f_w)], axis=1)
            b_temp = tf.stack([tf.ones_like(self.b), self.b], axis=0)
            self.logits = tf.matmul(f_w_temp, b_temp)
        else:
            self.logits = tf.nn.bias_add(tf.matmul(self.X, self.w), self.b)

        if self.cardinality == 2:
            self.logits = tf.squeeze(self.logits)

        # Define marginals op
        marginals_fn = tf.nn.softmax if self.cardinality > 2 else tf.nn.sigmoid
        self.marginals_op = marginals_fn(self.logits)

    def _build_training_ops(self, l1_penalty=0.0, l2_penalty=0.0, **kwargs):
        """
        Build training ops, setting loss and train ops
        
        :param l1_penalty: L1 reg. coefficient
        :param l2_penalty: L2 reg. coefficient
        """
        super(LogisticRegression, self)._build_training_ops()   
        
        # Add L1 and L2 penalties
        if l1_penalty > 0:
            self.loss += l1_penalty * tf.reduce_sum(tf.abs(self.w))
        if l2_penalty > 0:
            self.loss += l2_penalty * tf.nn.l2_loss(self.w)

    def _construct_feed_dict(self, X_b, Y_b, lr=0.01, **kwargs):
        return {self.X: X_b, self.Y: Y_b, self.lr: lr}

    def _check_input(self, X):
        if issparse(X):
            raise Exception("Sparse input matrix. Use SparseLogisticRegression")
        return X

    def _marginals_batch(self, X_test):
        X_test = self._check_input(X_test)
        return self.session.run(self.marginals_op, {self.X: X_test})

    def get_weights(self):
        """Get model weights and bias"""
        w, b = self.session.run([self.w, self.b])
        return np.ravel(w), np.ravel(b)


class SparseLogisticRegression(LogisticRegression):
    representation = False

    def _build_model(self, d=None, **kwargs):
        # Define sparse input placeholders + sparse tensors
        self.indices = tf.placeholder(tf.int64) 
        self.shape   = tf.placeholder(tf.int64, (2,))
        self.ids     = tf.placeholder(tf.int64)
        self.weights = tf.placeholder(tf.float32)
        sparse_ids   = tf.SparseTensor(self.indices, self.ids, self.shape)
        sparse_vals  = tf.SparseTensor(self.indices, self.weights, self.shape)
        self.Y = tf.placeholder(tf.float32, (None, self.cardinality)) \
            if self.cardinality > 2 else tf.placeholder(tf.float32, (None,))

        # Define parameters and logits
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)
        k = self.cardinality if self.cardinality > 2 else 1
        self.w = tf.Variable(tf.random_normal((d, k), stddev=SD, seed=s1))
        self.b = tf.Variable(tf.random_normal((k,), stddev=SD, seed=s2))
        
        if self.deterministic:
            # TODO: Implement for categorical as well...
            if self.cardinality > 2:
                raise NotImplementedError(
                    "Deterministic mode not implemented for categoricals.")

            # Try to make deterministic...
            f_w = tf.nn.embedding_lookup_sparse(params=self.w, 
                sp_ids=sparse_ids, sp_weights=sparse_vals, combiner=None)
            f_w_temp = tf.concat([f_w, tf.ones_like(f_w)], axis=1)
            b_temp = tf.stack([tf.ones_like(self.b), self.b], axis=0)
            self.logits = tf.matmul(f_w_temp, b_temp)
        else:
            z = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids,
                sp_weights=sparse_vals, combiner='sum')
            self.logits = tf.nn.bias_add(z, self.b)
        
        if self.cardinality == 2:
            self.logits = tf.squeeze(self.logits)

        # Define marginals op
        marginals_fn = tf.nn.softmax if self.cardinality > 2 else tf.nn.sigmoid
        self.marginals_op = marginals_fn(self.logits)

    def _check_input(self, X):
        if not issparse(X):
            msg = "Dense input matrix. Cast to sparse or use LogisticRegression"
            raise Exception(msg)
        return X.tocsr()

    def _batch_sparse_data(self, X):
        """
        Convert sparse batch matrix to sparse inputs for embedding lookup
        Notes: https://github.com/tensorflow/tensorflow/issues/342
        """
        if not issparse(X):
            raise Exception("Matrix X must be scipy.sparse type")
        X_lil = X.tolil()
        indices, ids, weights = [], [], []
        max_len = 0
        for i, (row, data) in enumerate(zip(X_lil.rows, X_lil.data)):
            # Dummy weight for all-zero row
            if len(row) == 0:
                indices.append((i, 0))
                ids.append(0)
                weights.append(0.0)
                continue
            # Update indices by position
            max_len = max(max_len, len(row))
            indices.extend((i, t) for t in range(len(row)))
            ids.extend(row)
            weights.extend(data)
        shape = (len(X_lil.rows), max_len)
        return indices, shape, ids, weights

    def _construct_feed_dict(self, X_b, Y_b, lr=0.01, **kwargs):
        indices, shape, ids, weights = self._batch_sparse_data(X_b)
        return {
            self.indices: indices,
            self.shape:   shape,
            self.ids:     ids,
            self.weights: weights,
            self.Y:       Y_b,
            self.lr:      lr
        }

    def _marginals_batch(self, X_test):
        X_test = self._check_input(X_test)
        if X_test.shape[0] == 0:
            return np.array([])
        indices, shape, ids, weights = self._batch_sparse_data(X_test)
        return self.session.run(self.marginals_op, {
            self.indices: indices,
            self.shape:   shape,
            self.ids:     ids,
            self.weights: weights
        })
