import numpy as np
import tensorflow as tf

from snorkel.learning import LabelBalancer, TFNoiseAwareModel
from utils import get_bi_rnn_output, SymbolTable
from time import time


class LSTMBase(TFNoiseAwareModel):

    representation = True

    def __init__(self, save_file=None, name='LSTMBase', seed=None, n_threads=4):
        """Base class for LSTM"""
        # Define metadata
        self.mx_len          = None # Max sentence length
        self.dim             = None # Embedding dimension
        self.n_v             = None # Vocabulary size
        self.lr              = None # Learning rate
        self.word_dict       = SymbolTable() # Symbol table for dictionary
        # Define input layers
        self.sentences       = None
        self.sentence_length = None
        self.marginals       = None
        self.keep_prob       = None
        self.seed            = seed
        # Super constructor
        super(LSTMBase, self).__init__(
            n_threads=n_threads, save_file=save_file, name=name
        )

    def _preprocess_data(self, candidates, extend):
        raise NotImplementedError()

    def _make_tensor(self, x):
        """Construct input tensor with padding"""
        batch_size = len(x)
        x_batch    = np.zeros((batch_size, self.mx_len), dtype=np.int32)
        len_batch  = np.zeros(batch_size, dtype=np.int32)
        for j, (token_ids, marginal) in enumerate(x):
            t               = min(len(token_ids), self.mx_len)
            x_batch[j, 0:t] = token_ids[0:t]
            len_batch[j]    = t
        return x_batch, len_batch

    def _embedding_init(self, s):
        return tf.random_normal((self.n_v-1, self.dim), seed=s)

    def _build(self):
        """Get feed forward step, loss function, and optimizer for LSTM"""
        # Define input layers
        self.sentences       = tf.placeholder(tf.int32, [None, None])
        self.sentence_length = tf.placeholder(tf.int32, [None])
        self.marginals       = tf.placeholder(tf.float32, [None])
        self.keep_prob       = tf.placeholder(tf.float32)
        # Embedding layer
        s         = self.seed
        emb_var   = tf.cast(tf.Variable(self._embedding_init(s)), tf.float32)
        s         += 1
        embedding = tf.concat([tf.zeros([1, self.dim]), emb_var], axis=0)
        inputs    = tf.nn.embedding_lookup(embedding, self.sentences)
        # Build RNN graph
        batch_size = tf.shape(self.sentences)[0]
        with tf.variable_scope("LSTM") as scope:
            tf.set_random_seed(s)
            s += 1
            # Build LSTM cells
            fw_cell = tf.contrib.rnn.BasicLSTMCell(self.dim)
            bw_cell = tf.contrib.rnn.BasicLSTMCell(self.dim)
            # Construct RNN
            initial_state_fw = fw_cell.zero_state(batch_size, tf.float32)
            initial_state_bw = bw_cell.zero_state(batch_size, tf.float32)
            rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, inputs, sequence_length=self.sentence_length,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                time_major=False               
            )
            s += 1
        # Get potentials
        potentials = get_bi_rnn_output(rnn_out, self.dim, self.sentence_length)
        # Compute activation
        potentials_dropout = tf.nn.dropout(potentials, self.keep_prob)
        W = tf.Variable(tf.random_normal((2*self.dim, 1), stddev=0.1, seed=s))
        s += 1
        b = tf.Variable(tf.random_normal((1,), stddev=0.1, seed=s))
        h = tf.squeeze(tf.matmul(potentials_dropout, W)) + b
        # Noise-aware loss
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.marginals, logits=h
        ))
        # Backprop trainer
        self.train_fn = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        # Get prediction
        self.prediction = tf.nn.sigmoid(h)

    def train(self, candidates, marginals, n_epochs=25, lr=0.01, dropout=0.5,
        dim=50, batch_size=64, max_sentence_length=None, rebalance=False,
        print_freq=5):
        """Train LSTM model
            @candidates: list Candidate objects for training
            @marginals:  array of marginal probabilities for each Candidate
            @n_epochs: number of training epochs
            @lr: learning rate
            @dropout: keep probability for dropout layer; if None, no dropout
            @dim: embedding dimension
            @batch_size: batch size for mini-batch SGD
            @max_sentence_length: maximum sentence length for candidates
            @rebalance: bool or fraction of positive examples desired
                        If True, defaults to standard 0.5 class balance.
                        If False, no class balancing.
            @print_freq: number of epochs after which to print status
        """
        verbose = print_freq > 0
        if verbose:
            print("[{0}] Dimension={1}  LR={2}".format(self.name, dim, lr))
            print("[{0}] Begin preprocessing".format(self.name))
            st = time()
        # Text preprocessing
        train_data = self._preprocess_data(candidates, extend=True)
        # Get training indices
        train_idxs = LabelBalancer(marginals).get_train_idxs(rebalance)
        x_train    = [train_data[j] for j in train_idxs]
        y_train    = np.ravel(marginals)[train_idxs]
        # Get max sentence size
        self.mx_len = max_sentence_length or max(len(x[0]) for x in x_train)
        # Build model
        self.dim = dim
        self.lr  = lr
        self.n_v = self.word_dict.len()
        self._build()
        # Run mini-batch SGD
        batch_size = min(batch_size, len(x_train))
        if verbose:
            print("[{0}] Preprocessing done ({1:.2f}s)".format(
                self.name, time() - st
            ))
            st = time()
            print("[{0}] Training model".format(self.name))
            print("[{0}] #examples={1}  #epochs={2}  batch size={3}".format(
                self.name, n, n_epochs, batch_size
            ))
        self.session.run(tf.global_variables_initializer())
        for t in range(n_epochs):
            epoch_loss = 0.0
            for i in range(0, len(x_train), batch_size):
                # Get batch tensors
                r          = min(n, i+batch_size)
                x_b, len_b = self._make_tensor(x_train[i:r])
                y_b        = y_train[i:r]
                # Run training step and evaluate loss function                  
                epoch_loss += self.session.run([self.loss, self.train_fn], {
                    self.sentences:       x_b,
                    self.sentence_length: len_b,
                    self.marginals:       y_b,
                    self.keep_prob:       dropout or 1.0,
                })[0]
            # Print training stats
            if verbose and (t % print_freq == 0 or t in [0, (n_epochs-1)]):
                print("[{0}] Epoch {1} ({2:.2f}s)\tAverage loss={3:.6f}".format(
                    self.name, t, time() - st, epoch_loss / n
                ))
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))

    def marginals(self, test_candidates):
        """Get likelihood of tagged sequences represented by test_candidates
            @test_candidates: list of lists representing test sentence
        """
        test_data = self._preprocess_data(test_candidates, extend=False)
        x, x_len = self._make_tensor(test_data)
        return np.ravel(self.session.run(self.prediction, {
            self.sentences:       x,
            self.sentence_length: x_len,
            self.keep_prob:       1.0,
        }))
