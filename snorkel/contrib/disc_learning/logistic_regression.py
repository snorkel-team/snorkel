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

    def _build(self, d=None):
        """
        Build network, loss, and prediction ops.
        @d: Number of features
        """
        # Build model
        if self.cardinality > 2:
            self._build_softmax(d)
        else:
            self._build_sigmoid(d)
        
        # Add L1 and L2 penalties
        if self.l1_penalty > 0:
            self.loss_op += self.l1_penalty * tf.reduce_sum(tf.abs(self.w))
        if self.l2_penalty > 0:
            self.loss_op += self.l2_penalty * tf.nn.l2_loss(self.w)
        
        # Build training op
        self.lr = tf.placeholder(tf.float32)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss_op)
        
        # Get nnz operation
        # TODO: Put this back in (need to refactor TFNoiseAwareModel.train)
        # self.nnz_op = tf.reduce_sum(tf.cast(
        #     tf.not_equal(self.w, tf.constant(0, tf.float32)), tf.int32))

    def _build_sigmoid(self, d):
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)

        # Define inputs and variables
        self.X = tf.placeholder(tf.float32, (None, d))
        self.Y = tf.placeholder(tf.float32, (None,))
        self.w = tf.Variable(tf.random_normal((d, 1), stddev=SD, seed=s1))
        self.b = tf.Variable(tf.random_normal((1,), stddev=SD, seed=s2))
        h = tf.squeeze(tf.nn.bias_add(tf.matmul(self.X, self.w), self.b))
        
        # Noise-aware loss op
        self.loss_op = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=h, labels=self.Y))
        
        # Get prediction op
        self.prediction_op = tf.nn.sigmoid(h)

    def _build_softmax(self, d):
        """Build for the categorical setting."""
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)

        # Define inputs and variables
        self.X = tf.placeholder(tf.float32, (None, d))
        self.Y = tf.placeholder(tf.float32, (None, self.cardinality))
        self.w = tf.Variable(tf.random_normal((d, self.cardinality), 
            stddev=SD, seed=s1))
        self.b = tf.Variable(
            tf.random_normal((self.cardinality,), stddev=SD, seed=s2))
        h = tf.nn.bias_add(tf.matmul(self.X, self.w), self.b)
        
        # Noise-aware loss op
        self.loss_op = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=self.Y))

        # Get prediction op
        self.prediction_op = tf.nn.softmax(h)

    def _construct_feed_dict(self, X_b, Y_b, lr=0.01, **kwargs):
        return {self.X: X_b, self.Y: Y_b, self.lr: lr}

    def _check_input(self, X):
        if issparse(X):
            raise Exception("Sparse input matrix. Use SparseLogisticRegression")
        return X

    def train(self, X_train, Y_train, n_epochs=10, lr=0.01, batch_size=100,
        l1_penalty=0.0, l2_penalty=0.0, **kwargs):
        """
        Train elastic net logistic regression model using TensorFlow
        @l1_penalty: l1 regularization strength
        @l2_penalty: l2 regularization strength
        """
        self._check_input(X_train)

        # We set these here rather then passing them through build because they
        # are not model params that need to be saved / reloaded
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        # Train model
        super(LogisticRegression, self).train(X_train, Y_train, 
            d=X_train.shape[1], n_epochs=n_epochs, lr=lr, batch_size=batch_size,
            **kwargs)

    def marginals(self, X_test):
        X_test = self._check_input(X_test)
        return self.session.run(self.prediction_op, {self.X: X_test})

    def get_weights(self):
        """Get model weights and bias"""
        w, b = self.session.run([self.w, self.b])
        return np.ravel(w), np.ravel(b)


class SparseLogisticRegression(LogisticRegression):
    representation = False

    def _build(self, d=None):
        # Define sparse input placeholders + sparse tensors
        self.indices = tf.placeholder(tf.int64) 
        self.shape   = tf.placeholder(tf.int64, (2,))
        self.ids     = tf.placeholder(tf.int64)
        self.weights = tf.placeholder(tf.float32)
        sparse_ids   = tf.SparseTensor(self.indices, self.ids, self.shape)
        sparse_vals  = tf.SparseTensor(self.indices, self.weights, self.shape)

        # Build network, loss, and prediction ops
        if self.cardinality > 2:
            self._build_softmax(d, sparse_ids, sparse_vals)
        else:
            self._build_sigmoid(d, sparse_ids, sparse_vals)

        # Add L1 and L2 penalties
        self.loss_op += self.l1_penalty * tf.reduce_sum(tf.abs(self.w))
        self.loss_op += self.l2_penalty * tf.nn.l2_loss(self.w)
        
        # Build training and save param ops
        self.lr = tf.placeholder(tf.float32)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss_op)
        
        # Get nnz operation
        # TODO: Put this back in (need to refactor TFNoiseAwareModel.train)
        # self.nnz = tf.reduce_sum(tf.cast(
        #     tf.not_equal(self.w, tf.constant(0, tf.float32)), tf.int32))

    def _build_sigmoid(self, d, sparse_ids, sparse_vals):
        """Build network for the binary setting."""
        self.Y = tf.placeholder(tf.float32, (None,))
        
        # Define model params
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)
        self.w = tf.Variable(tf.random_normal((d, 1), stddev=SD, seed=s1))
        self.b = tf.Variable(tf.random_normal((1,), stddev=SD, seed=s2))
        z = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids,
            sp_weights=sparse_vals, combiner='sum')
        h = tf.squeeze(tf.nn.bias_add(z, self.b))
        
        # Noise-aware loss and prediction ops
        self.loss_op = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=h, labels=self.Y))
        self.prediction_op = tf.nn.sigmoid(h)

    def _build_softmax(self, d, sparse_ids, sparse_vals):
        """Build network for the categorical setting."""
        self.Y = tf.placeholder(tf.float32, (None, self.cardinality))
        
        # Define model params
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)
        self.w = tf.Variable(
            tf.random_normal((d, self.cardinality), stddev=SD, seed=s1))
        self.b = tf.Variable(
            tf.random_normal((self.cardinality,), stddev=SD, seed=s2))
        z = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids,
            sp_weights=sparse_vals, combiner='sum')
        h = tf.nn.bias_add(z, self.b)
        
        # Noise-aware loss and prediction ops
        self.loss_op = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=self.Y))
        self.prediction_op = tf.nn.softmax(h)

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
            indices.extend((i, t) for t in xrange(len(row)))
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

    def marginals(self, X_test):
        X_test = self._check_input(X_test)
        if X_test.shape[0] == 0:
            return np.array([])
        indices, shape, ids, weights = self._batch_sparse_data(X_test)
        return self.session.run(self.prediction_op, {
            self.indices: indices,
            self.shape:   shape,
            self.ids:     ids,
            self.weights: weights
        })


if __name__ == '__main__':
    # Generate data
    n, d = 50, 5
    X = np.round(np.random.rand(n, d))
    y = np.random.rand(n)
    # Test logistic regression
    F = LogisticRegression()
    F.train(X, y, 3, 0.01, 10, 1e-4, 1e-4, 1, True, 1701)
    # Test sparse logistic regression
    X_sp = csr_matrix(X)
    F = SparseLogisticRegression()
    F.train(X_sp, y, 3, 0.01, 10, 1e-4, 1e-4, 1, True, 1701)

