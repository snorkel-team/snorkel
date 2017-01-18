import cPickle
import numpy as np
import tensorflow as tf

from logistic_regression import LogisticRegression
from scipy.sparse import issparse
from time import time
from utils import get_train_idxs


class SparseLogisticRegression(LogisticRegression):

    def __init__(self, save_file=None, name='SparseLR'):
        """Sparse noise-aware logistic regression in TensorFlow"""
        self.indices = None
        self.shape   = None
        self.ids     = None
        self.weights = None
        super(SparseLogisticRegression, self).__init__(
            save_file=save_file, name=name
        )

    def _build(self):
        # Define input placeholders
        self.indices = tf.placeholder(tf.int64) 
        self.shape   = tf.placeholder(tf.int64, (2,))
        self.ids     = tf.placeholder(tf.int64)
        self.weights = tf.placeholder(tf.float32)
        self.Y       = tf.placeholder(tf.float32, (None, 1))
        # Define training variables
        sparse_ids = tf.SparseTensor(self.indices, self.ids, self.shape)
        sparse_vals = tf.SparseTensor(self.indices, self.weights, self.shape)
        w = tf.Variable(tf.random_normal((self.d, 1), mean=0, stddev=0.01))
        b = tf.Variable(tf.random_normal((1, 1), mean=0, stddev=0.01))
        z = tf.embedding_lookup_sparse(
            params=w, sp_ids=sparse_ids, sp_weights=sparse_vals, combiner='sum'
        )
        h = tf.add(z, b)
        # Build model
        self.loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(h, self.Y)
        )
        self.train_fn = tf.train.ProximalGradientDescentOptimizer(
            learning_rate=tf.cast(self.lr, dtype=tf.float32),
            l1_regularization_strength=tf.cast(self.l1_penalty, tf.float32),
            l2_regularization_strength=tf.cast(self.l2_penalty, tf.float32),
        ).minimize(self.loss)
        self.prediction = tf.nn.sigmoid(h)
        self.save_dict = {'w': w, 'b': b}

    def _batch_sparse_data(self, X):
        """Convert CSR batch matrix to sparse inputs for embedding lookup"""
        X_lil = X.tolil()
        indices, ids, weights = [], [], []
        max_len = 0
        for i, (row, data) in enumerate(zip(X_lil.rows, X_lil.data)):
            max_len = max(max_len, len(row))
            indices.extend((i, t) for t in xrange(len(row)))
            ids.extend(row)
            weights.extend(data)
        shape = (len(X_lil.rows), max_len)
        return indices, shape, ids, weights

    def _run_batch(self, X_train, y_train, i, r):
        """Run a single batch update"""
        # Get batch sparse tensor data
        indices, shape, ids, weights = self._batch_sparse_data(X_train[i:r, :])
        y_batch = y_train[i:r].reshape((r-i, 1))
        # Run training step and evaluate loss function                  
        return self.session.run([self.loss, self.train_fn], {
            self.indices: indices,
            self.shape:   shape,
            self.ids:     ids,
            self.weights: weights,
            self.Y:       y_batch,
        })[0]

    def marginals(self, X_test):
        indices, shape, ids, weights = self._batch_sparse_data(X_test)
        return np.ravel(self.session.run([self.prediction], {
            self.indices: indices,
            self.shape:   shape,
            self.ids:     ids,
            self.weights: weights,
        })
