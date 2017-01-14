import cPickle
import numpy as np
import tensorflow as tf

from disc_learning import TFNoiseAwareModel
from time import time


def LogisticRegression(TFNoiseAwareModel):

    def __init__(self, name='LR'):
        """Noise-aware logistic regression in TensorFlow"""
        super(LogisticRegression, self).__init__(name)
        self.d = None
        self.X = None

    def _build(self):
        self.X = tf.placeholder(tf.float32, (None, self.d))
        Y = tf.placeholder(tf.float32, (None, 1))
        w = tf.Variable(tf.random_normal((self.d, 1), mean=0, stddev=2.0))
        b = tf.Variable(tf.random_normal((1, 1), mean=0, stddev=2.0))
        h = tf.add(tf.matmul(self.X, w), b)
        # Build model
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(h, Y)
        )
        self.train_fn = tf.train.ProximalGradientDescentOptimizer(
            learning_rate=self.lr,
            l1_regularization_strength=self.l1_penalty,
            l2_regularization_strength=self.l2_penalty,
        ).minimize(self.loss)
        self.prediction = tf.nn.sigmoid(self.h)

    def train(self, X, training_marginals, n_epochs=10, lr=0.01, batch_size=100,
        l1_penalty=0.0, l2_penalty=0.0, print_freq=5, model_name=None):
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
        # Run mini-batch SGD
        n = X.shape[0]
        batch_size = min(batch_size, n)
        if verbose:
            st = time()
            print("[{0}] Training model  #epochs={1}  batch={2}".format(
                self.name, n_epochs, batch_size
            ))
        for t in xrange(n_epochs):
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                # Get batch tensors
                x_batch = X[i:i+batch_size, :]
                y_batch = training_marginals[i:i+batch_size]
                # Run training step and evaluate cost function                  
                epoch_loss += self.session.run([self.cost, self.train_fn], {
                    self.X: x_batch,
                    self.Y: y_batch,
                })[0]
            # Print training stats
            if verbose and ((t+1) % n_print == 0 or t == (n_epochs-1)):
                print("[{0}] Epoch {1} ({2.2f}s)\tLoss={3:.6f}".format(
                    self.name, t+1, time() - st, epoch_loss
                ))
        # Save model
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))
        self.save(model_name, verbose=verbose)

    def marginals(self, X):
        return np.ravel(self.session.run([prediction], {self.X: X}))

    def save_info(self, model_name):
        with open('{0}.info'.format(model_name), 'wb') as f:
            cPickle.dump(self.d, f)

    def load_info(self, model_name):
        with open('{0}.info'.format(model_name), 'rb') as f:
            self.d = cPickle.load(f)
