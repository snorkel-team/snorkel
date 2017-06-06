import numpy as np
import tensorflow as tf

from snorkel.learning.disc_learning import TFNoiseAwareModel
from scipy.sparse import csr_matrix, issparse
from time import time
from six.moves.cPickle import dump, load
from snorkel.learning.utils import LabelBalancer, get_cardinality

SD = 0.1


class LogisticRegression(TFNoiseAwareModel):

    def __init__(self, save_file=None, name='LR', n_threads=None):
        """Noise-aware logistic regression in TensorFlow"""
        self.d          = None
        self.X          = None
        self.Y          = None
        self.lr         = None
        self.l1_penalty = None
        self.l2_penalty = None
        self.w          = None
        self.b          = None
        self.nnz        = None
        super(LogisticRegression, self).__init__(
            save_file=save_file, name=name, n_threads=n_threads
        )

    def _build(self):
        # Build network, loss, and prediction ops
        if self.k > 2:
            self._build_softmax()
        else:
            self._build_sigmoid()
        
        # Add L1 and L2 penalties
        self.loss += self.l1_penalty * tf.reduce_sum(tf.abs(self.w))
        self.loss += self.l2_penalty * tf.nn.l2_loss(self.w)
        
        # Build model
        self.train_fn = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        # Save operation
        self.save_dict = {'w': self.w, 'b': self.b}
        
        # Get nnz operation
        self.nnz = tf.reduce_sum(tf.cast(
            tf.not_equal(self.w, tf.constant(0, tf.float32)), tf.int32
        ))

    def _build_sigmoid(self):
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)

        # Define inputs and variables
        self.X = tf.placeholder(tf.float32, (None, self.d))
        self.Y = tf.placeholder(tf.float32, (None,))
        self.w = tf.Variable(tf.random_normal((self.d, 1), stddev=SD, seed=s1))
        self.b = tf.Variable(tf.random_normal((1,), stddev=SD, seed=s2))
        h = tf.squeeze(tf.nn.bias_add(tf.matmul(self.X, self.w), self.b))
        
        # Noise-aware loss op
        self.loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=h, labels=self.Y)
        )
        
        # Get prediction op
        self.prediction = tf.nn.sigmoid(h)

    def _build_softmax(self):
        """Build for the categorical setting."""
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)

        # Define inputs and variables
        self.X = tf.placeholder(tf.float32, (None, self.d))
        self.Y = tf.placeholder(tf.float32, (None, self.k))
        self.w = tf.Variable(tf.random_normal((self.d, self.k), stddev=SD, 
            seed=s1))
        self.b = tf.Variable(tf.random_normal((self.k,), stddev=SD, seed=s2))
        h = tf.nn.bias_add(tf.matmul(self.X, self.w), self.b)
        
        # Noise-aware loss op
        self.loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=self.Y)
        )

        # Get prediction op
        self.prediction = tf.nn.softmax(h)

    def _check_input(self, X):
        if issparse(X):
            raise Exception("Sparse input matrix. Use SparseLogisticRegression")
        return X

    def _run_batch(self, X_train, y_train, i, r, *args):
        """Run a single batch update"""
        # Get batch tensors
        sparse  = issparse(X_train)
        x_batch = X_train[i:r, :].todense() if sparse else X_train[i:r, :]
        y_batch = y_train[i:r]
        # Run training step and evaluate loss function                  
        return self.session.run([self.loss, self.train_fn, self.nnz], {
            self.X: x_batch, self.Y: y_batch,
        })

    def train(self, X, training_marginals, n_epochs=10, lr=0.01,
        batch_size=100, l1_penalty=0.0, l2_penalty=0.0, print_freq=5,
        rebalance=False, seed=None):
        """Train elastic net logistic regression model using TensorFlow
            @X: SciPy or NumPy feature matrix
            @training_marginals: N x K array of marginals for examples in X,
                where K \in {0,1,2} for binary and > 2 for categorical
            @n_epochs: number of training epochs
            @lr: learning rate
            @batch_size: batch size for mini-batch SGD
            @l1_penalty: l1 regularization strength
            @l2_penalty: l2 regularization strength
            @print_freq: number of epochs after which to print status
            @rebalance: bool or fraction of positive examples desired
                        If True, defaults to standard 0.5 class balance.
                        If False, no class balancing.
        """
        # Make sure training marginals are a numpy array + get cardinality
        training_marginals, self.k = get_cardinality(training_marginals)

        # Build model
        X = self._check_input(X)
        verbose = print_freq > 0
        if verbose:
            print("[{0}] lr={1} l1={2} l2={3}".format(
                self.name, lr, l1_penalty, l2_penalty
            ))
            print("[{0}] Building model".format(self.name))
        self.d          = X.shape[1]
        self.lr         = lr
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.seed       = seed
        self._build()
        # Get training indices
        # Note: Currently we only do label balancing for binary setting
        if self.k == 2:
            train_idxs = LabelBalancer(training_marginals).\
                get_train_idxs(rebalance)
            X_train = X[train_idxs, :]
            y_train = np.ravel(training_marginals)[train_idxs]
        else:
            X_train = X
            y_train = training_marginals
        # Run mini-batch SGD
        n = X_train.shape[0]
        batch_size = min(batch_size, n)
        nnz = 0
        if verbose:
            st = time()
            print("[{0}] Training model".format(self.name))
            print("[{0}] #examples={1}  #epochs={2}  batch size={3}".format(
                self.name, n, n_epochs, batch_size
            ))
        self.session.run(tf.global_variables_initializer())
        for t in xrange(n_epochs):
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                r = min(n-1, i+batch_size)
                loss, _, nnz = self._run_batch(X_train, y_train, i, r, nnz)
                epoch_loss += loss
            # Print training stats
            if verbose and (t % print_freq == 0 or t in [0, (n_epochs-1)]):
                msg = "[{0}] Epoch {1} ({2:.2f}s)\tAvg. loss={3:.6f}\tNNZ={4}"
                print(msg.format(self.name, t, time()-st, epoch_loss/n, nnz))
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))

    def marginals(self, X_test):
        X_test = self._check_input(X_test)
        return self.session.run(self.prediction, {self.X: X_test})

    def get_weights(self):
        """Get model weights and bias"""
        w, b = self.session.run([self.w, self.b])
        return np.ravel(w), np.ravel(b)

    def save_info(self, model_name):
        with open('{0}.info'.format(model_name), 'wb') as f:
            dump((self.d, self.lr, self.l1_penalty, self.l2_penalty), f)

    def load_info(self, model_name):
        with open('{0}.info'.format(model_name), 'rb') as f:
            self.d, self.lr, self.l1_penalty, self.l2_penalty = load(f)


class SparseLogisticRegression(LogisticRegression):

    def __init__(self, save_file=None, name='SparseLR', n_threads=None):
        """Sparse noise-aware logistic regression in TensorFlow"""
        self.indices = None
        self.shape   = None
        self.ids     = None
        self.weights = None
        super(SparseLogisticRegression, self).__init__(
            save_file=save_file, name=name, n_threads=n_threads
        )

    def _build_sigmoid(self, sparse_ids, sparse_vals):
        """Build network for the binary setting."""
        self.Y = tf.placeholder(tf.float32, (None,))
        
        # Define model params
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)
        self.w = tf.Variable(tf.random_normal((self.d, 1), stddev=SD, seed=s1))
        self.b = tf.Variable(tf.random_normal((1,), stddev=SD, seed=s2))
        z = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids,
            sp_weights=sparse_vals, combiner='sum')
        h = tf.squeeze(tf.nn.bias_add(z, self.b))
        
        # Noise-aware loss and prediction ops
        self.loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=h, labels=self.Y)
        )
        self.prediction = tf.nn.sigmoid(h)

    def _build_softmax(self, sparse_ids, sparse_vals):
        """Build network for the categorical setting."""
        self.Y = tf.placeholder(tf.float32, (None, self.k))
        
        # Define model params
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)
        self.w = tf.Variable(tf.random_normal((self.d, self.k), stddev=SD, 
            seed=s1))
        self.b = tf.Variable(tf.random_normal((self.k,), stddev=SD, seed=s2))
        z = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids,
            sp_weights=sparse_vals, combiner='sum')
        h = tf.nn.bias_add(z, self.b)
        
        # Noise-aware loss and prediction ops
        self.loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=self.Y)
        )
        self.prediction = tf.nn.softmax(h)

    def _build(self):
        # Define sparse input placeholders + sparse tensors
        self.indices = tf.placeholder(tf.int64) 
        self.shape   = tf.placeholder(tf.int64, (2,))
        self.ids     = tf.placeholder(tf.int64)
        self.weights = tf.placeholder(tf.float32)
        sparse_ids   = tf.SparseTensor(self.indices, self.ids, self.shape)
        sparse_vals  = tf.SparseTensor(self.indices, self.weights, self.shape)

        # Build network, loss, and prediction ops
        if self.k > 2:
            self._build_softmax(sparse_ids, sparse_vals)
        else:
            self._build_sigmoid(sparse_ids, sparse_vals)

        # Add L1 and L2 penalties
        self.loss += self.l1_penalty * tf.reduce_sum(tf.abs(self.w))
        self.loss += self.l2_penalty * tf.nn.l2_loss(self.w)
        
        # Build training and save param ops
        self.train_fn = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.save_dict = {'w': self.w, 'b': self.b}
        
        # Get nnz operation
        self.nnz = tf.reduce_sum(tf.cast(
            tf.not_equal(self.w, tf.constant(0, tf.float32)), tf.int32
        ))

    def _check_input(self, X):
        if not issparse(X):
            msg = "Dense input matrix. Cast to sparse or use LogisticRegression"
            raise Exception(msg)
        return X.tocsr()

    def _batch_sparse_data(self, X):
        """Convert sparse batch matrix to sparse inputs for embedding lookup
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

    def _run_batch(self, X_train, y_train, i, r, last_nnz):
        """Run a single batch update"""
        # Get batch sparse tensor data
        indices, shape, ids, weights = self._batch_sparse_data(X_train[i:r, :])
        y_batch = y_train[i:r]
        # Run training step and evaluate loss function
        if len(indices) == 0:
            return 0.0, None, last_nnz   
        return self.session.run([self.loss, self.train_fn, self.nnz], {
            self.indices: indices,
            self.shape:   shape,
            self.ids:     ids,
            self.weights: weights,
            self.Y:       y_batch,
        })

    def marginals(self, X_test):
        X_test = self._check_input(X_test)
        if X_test.shape[0] == 0:
            return np.array([])
        indices, shape, ids, weights = self._batch_sparse_data(X_test)
        return self.session.run(self.prediction, {
            self.indices: indices,
            self.shape:   shape,
            self.ids:     ids,
            self.weights: weights,
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

