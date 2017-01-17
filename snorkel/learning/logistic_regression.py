import cPickle
import numpy as np
import tensorflow as tf

from disc_learning import TFNoiseAwareModel
from scipy.sparse import issparse
from time import time
from utils import get_train_idxs


class LogisticRegression(TFNoiseAwareModel):

    def __init__(self, save_file=None, name='LR'):
        """Noise-aware logistic regression in TensorFlow"""
        self.d          = None
        self.X          = None
        self.lr         = None
        self.l1_penalty = None
        self.l2_penalty = None
        super(LogisticRegression, self).__init__(save_file=save_file, name=name)

    def _build(self):
        # TODO: switch to sparse variables
        self.X = tf.placeholder(tf.float32, (None, self.d))
        self.Y = tf.placeholder(tf.float32, (None, 1))
        w = tf.Variable(tf.random_normal((self.d, 1), mean=0, stddev=0.01))
        b = tf.Variable(tf.random_normal((1, 1), mean=0, stddev=0.01))
        h = tf.add(tf.matmul(self.X, w), b)
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

    def train(self, X, training_marginals, n_epochs=10, lr=0.01,
        batch_size=100, l1_penalty=0.0, l2_penalty=0.0, print_freq=5,
        rebalance=False):
        """Train elastic net logistic regression model using TensorFlow
            @X: SciPy or NumPy feature matrix
            @training_marginals: array of marginals for examples in X
            @n_epochs: number of training epochs
            @lr: learning rate
            @batch_size: batch size for mini-batch SGD
            @l1_penalty: l1 regularization strength
            @l2_penalty: l2 regularization strength
            @print_freq: number of epochs after which to print status
            @rebalance: rebalance training examples?
        """
        # Build model
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
        self._build()
        # Get training indices
        train_idxs = get_train_idxs(training_marginals, rebalance=rebalance)
        X_train = X[train_idxs, :]
        y_train = np.ravel(training_marginals)[train_idxs]
        # Run mini-batch SGD
        n = X_train.shape[0]
        batch_size = min(batch_size, n)
        if verbose:
            st = time()
            print("[{0}] Training model  #epochs={1}  batch={2}".format(
                self.name, n_epochs, batch_size
            ))
        self.session.run(tf.global_variables_initializer())
        for t in xrange(n_epochs):
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                # Get batch tensors
                r = min(n-1, i+batch_size)
                x_batch = X_train[i:r, :].todense()
                y_batch = y_train[i:r]
                y_batch = y_batch.reshape((len(y_batch), 1))
                # Run training step and evaluate loss function                  
                epoch_loss += self.session.run([self.loss, self.train_fn], {
                    self.X: x_batch,
                    self.Y: y_batch,
                })[0]
            # Print training stats
            if verbose and (t % print_freq == 0 or t in [0, (n_epochs-1)]):
                print("[{0}] Epoch {1} ({2:.2f}s)\tAverage loss={3:.6f}".format(
                    self.name, t, time() - st, epoch_loss / n
                ))
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))

    def marginals(self, X_test):
        X = X_test.todense() if issparse(X_test) else X_test
        return np.ravel(self.session.run([self.prediction], {self.X: X}))

    def save_info(self, model_name):
        with open('{0}.info'.format(model_name), 'wb') as f:
            cPickle.dump((self.d, self.lr, self.l1_penalty, self.l2_penalty), f)

    def load_info(self, model_name):
        with open('{0}.info'.format(model_name), 'rb') as f:
            self.d, self.lr, self.l1_penalty, self.l2_penalty = cPickle.load(f)
