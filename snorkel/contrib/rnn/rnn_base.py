import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import warnings

from snorkel.learning import LabelBalancer, TFNoiseAwareModel, get_cardinality
from utils import f1_score, get_bi_rnn_output, SymbolTable
from time import time


SD = 0.1


class RNNBase(TFNoiseAwareModel):

    representation = True

    def __init__(self, save_file=None, name='RNNBase', seed=None, n_threads=4):
        """Base class for bidirectional RNN"""
        # Define metadata
        self.mx_len    = None # Max sentence length
        self.dim       = None # Embedding dimension
        self.n_v       = None # Vocabulary size
        self.lr        = None # Learning rate
        self.attn      = None # Attention window
        self.cell      = None # RNN cell type
        self.word_dict = SymbolTable() # Symbol table for dictionary
        # Define input layers
        self.sentences        = None
        self.sentence_lengths = None
        self.train_marginals  = None
        self.keep_prob        = None
        self.seed             = seed
        # Super constructor
        super(RNNBase, self).__init__(
            n_threads=n_threads, save_file=save_file, name=name
        )

    def _preprocess_data(self, candidates, extend):
        """Build @self.word_dict to encode and process data for extraction
            Return list of encoded sentences and list of last index of arguments
        """
        raise NotImplementedError()

    def _check_max_sentence_length(self, ends):
        """Check that extraction arguments are within @self.mx_len"""
        mx = self.mx_len
        for i, end in enumerate(ends):
            if end >= mx:
                w = "Candidate {0} has argument past max length for model:"
                info = "[arg ends at index {0}; max len {1}]".format(end, mx)
                warnings.warn('\t'.join([w.format(i), info]))

    def _make_tensor(self, x):
        """Construct input tensor with padding
            Builds a matrix of symbols corresponding to @self.word_dict for the
            current batch and an array of true sentence lengths
        """
        batch_size = len(x)
        x_batch    = np.zeros((batch_size, self.mx_len), dtype=np.int32)
        len_batch  = np.zeros(batch_size, dtype=np.int32)
        for j, token_ids in enumerate(x):
            t               = min(len(token_ids), self.mx_len)
            x_batch[j, 0:t] = token_ids[0:t]
            len_batch[j]    = t
        return x_batch, len_batch

    def _embedding_init(self, s):
        """Random initialization for embedding table"""
        return tf.random_normal((self.n_v-1, self.dim), stddev=SD, seed=s)

    def _build(self):
        """Get feed forward step, loss function, and optimizer for RNN"""
        # Define input layers
        self.sentences        = tf.placeholder(tf.int32, [None, None])
        self.sentence_lengths = tf.placeholder(tf.int32, [None])
        self.keep_prob        = tf.placeholder(tf.float32)
        # Seeds
        s = self.seed
        s1, s2, s3, s4 = [None] * 4 if s is None else [s+i for i in range(4)]
        # Embedding layer
        emb_var   = tf.Variable(self._embedding_init(s1))
        embedding = tf.concat([tf.zeros([1, self.dim]), emb_var], axis=0)
        inputs    = tf.nn.embedding_lookup(embedding, self.sentences)
        # Build RNN graph
        batch_size = tf.shape(self.sentences)[0]
        rand_name  = "RNN_{0}".format(random.randint(0, 1e12)) # Obscene hack
        init = tf.contrib.layers.xavier_initializer(seed=s2)
        with tf.variable_scope(rand_name, reuse=False, initializer=init):
            # Build RNN cells
            fw_cell = self.cell(self.dim)
            bw_cell = self.cell(self.dim)
            # Add attention if needed
            if self.attn:
                fw_cell = rnn.AttentionCellWrapper(
                    fw_cell, self.attn, state_is_tuple=True
                )
                bw_cell = rnn.AttentionCellWrapper(
                    bw_cell, self.attn, state_is_tuple=True
                )
            # Construct RNN
            initial_state_fw = fw_cell.zero_state(batch_size, tf.float32)
            initial_state_bw = bw_cell.zero_state(batch_size, tf.float32)
            rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, inputs,
                sequence_length=self.sentence_lengths,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                time_major=False               
            )
        # Get potentials
        potentials = get_bi_rnn_output(rnn_out, self.dim, self.sentence_lengths)
        
        # Compute activation
        potentials_dropout = tf.nn.dropout(potentials, self.keep_prob, seed=s3)
        if self.k > 2:
            self._build_softmax(potentials_dropout, s4)
        else:
            self._build_sigmoid(potentials_dropout, s4)

        # Backprop trainer
        self.train_fn = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _build_sigmoid(self, potentials, seed):
        self.train_marginals = tf.placeholder(tf.float32, [None])
        W = tf.Variable(tf.random_normal((2*self.dim, 1), stddev=SD, seed=seed))
        b = tf.Variable(0., dtype=tf.float32)
        h_dropout = tf.squeeze(tf.matmul(potentials, W)) + b
        # Noise-aware loss
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.train_marginals, logits=h_dropout
        ))
        # Get prediction
        self.prediction = tf.nn.sigmoid(h_dropout)

    def _build_softmax(self, potentials, seed):
        self.train_marginals = tf.placeholder(tf.float32, [None, self.k])
        W = tf.Variable(tf.random_normal((2*self.dim, self.k), stddev=SD, 
            seed=seed))
        b = tf.Variable(np.zeros(self.k), dtype=tf.float32)
        h_dropout = tf.matmul(potentials, W) + b
        # Noise-aware loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.train_marginals, logits=h_dropout
        ))
        # Get prediction
        self.prediction = tf.nn.softmax(h_dropout)

    def train(self, candidates, marginals, n_epochs=25, lr=0.01, dropout=0.5,
        dim=50, attn_window=None, cell_type=rnn.BasicLSTMCell, batch_size=256,
        max_sentence_length=None, rebalance=False, dev_candidates=None,
        dev_labels=None, print_freq=5):
        """Train bidirectional RNN model for binary classification
            @candidates: list of Candidate objects for training
            @marginals: array of marginal probabilities for each Candidate
            @n_epochs: number of training epochs
            @lr: learning rate
            @dropout: keep probability for dropout layer (no dropout if None)
            @dim: embedding dimension
            @attn_window: attention window length (no attention if 0 or None)
            @cell_type: subclass of tensorflow.python.ops.rnn_cell_impl._RNNCell
            @batch_size: batch size for mini-batch SGD
            @max_sentence_length: maximum sentence length for candidates
            @rebalance: bool or fraction of positive examples for training
                        - if True, defaults to standard 0.5 class balance
                        - if False, no class balancing
            @dev_candidates: list of Candidate objects for evaluation
            @dev_labels: array of labels for each dev Candidate
            @print_freq: number of epochs after which to print status
        """
        marginals, self.k = get_cardinality(marginals)
        verbose = print_freq > 0
        if verbose:
            print("[{0}] Dimension={1}  LR={2}".format(self.name, dim, lr))
            print("[{0}] Begin preprocessing".format(self.name))
            st = time()
        # Text preprocessing
        train_data, ends = self._preprocess_data(candidates, extend=True)
        # Get training indices
        np.random.seed(self.seed)
        # Get training indices
        # Note: Currently we only do label balancing for binary setting
        if self.k == 2:
            train_idxs = LabelBalancer(marginals).get_train_idxs(rebalance)
            x_train    = [train_data[j] for j in train_idxs]
            y_train    = np.ravel(marginals)[train_idxs]
        else:
            x_train = train_data
            y_train = marginals
        # Get max sentence size
        self.mx_len = max_sentence_length or max(len(x) for x in x_train)
        self._check_max_sentence_length(ends)
        # Build model
        self.dim  = dim
        self.lr   = lr
        self.n_v  = self.word_dict.len()
        self.attn = attn_window
        self.cell = cell_type
        self._build()
        # Get dev data
        dev_data, dev_gold = None, None
        if dev_candidates is not None and dev_labels is not None:
            dev_data, _ = self._preprocess_data(dev_candidates, extend=False)
            dev_gold = np.ravel(dev_labels)
            if not ((dev_gold >= 0).all() and (dev_gold <= 1).all()):
                raise Exception("Dev labels should be in [0, 1]")
            print("[{0}] Loaded {1} candidates for evaluation".format(
                self.name, len(dev_data)
            ))
        # Run mini-batch SGD
        n = len(x_train)
        batch_size = min(batch_size, n)
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
            epoch_loss = []
            for i in range(0, n, batch_size):
                # Get batch tensors
                x_b, len_b = self._make_tensor(x_train[i:i+batch_size])
                y_b        = y_train[i:i+batch_size]
                # Run training step and evaluate loss function                  
                epoch_loss.append(self.session.run([self.loss, self.train_fn], {
                    self.sentences:        x_b,
                    self.sentence_lengths: len_b,
                    self.train_marginals:  y_b,
                    self.keep_prob:        dropout or 1.0,
                })[0])
            # Print training stats
            if verbose and (t % print_freq == 0 or t in [0, (n_epochs-1)]):
                msg = "[{0}] Epoch {1} ({2:.2f}s)\tAverage loss={3:.6f}".format(
                    self.name, t, time() - st, np.mean(epoch_loss)
                )
                if dev_data is not None:
                    dev_p    = self._marginals_preprocessed(dev_data)
                    f1, _, _ = f1_score(dev_p, dev_gold)
                    msg     += '\tDev F1={0:.2f}'.format(100. * f1)
                print(msg)
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))

    def _marginals_preprocessed(self, test_data):
        """Get marginals from preprocessed data"""
        x, x_len = self._make_tensor(test_data)
        return self.session.run(self.prediction, {
            self.sentences:        x,
            self.sentence_lengths: x_len,
            self.keep_prob:        1.0,
        })

    def marginals(self, test_candidates):
        """Get likelihood of tagged sequences represented by test_candidates
            @test_candidates: list of lists representing test sentence
        """
        test_data, ends = self._preprocess_data(test_candidates, extend=False)
        self._check_max_sentence_length(ends)
        return self._marginals_preprocessed(test_data)
