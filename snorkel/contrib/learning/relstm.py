import cPickle
import numpy as np
import tensorflow as tf

from snorkel.learning import get_train_idxs, TFNoiseAwareModel
from time import time


class reLSTM(TFNoiseAwareModel):

    def __init__(self, save_file=None, name='reLSTM'):
        """LSTM for relation extraction"""
        # Define metadata
        self.mx_len          = None # Max sentence length
        self.dim             = None # Embedding dimension
        self.n_v             = None # Vocabulary size
        self.lr              = None # Learning rate
        self.dropout         = None # Dropout rate
        self.tokens          = None # Token type for sentences (e.g. lemmas)
        self.word_dict       = SymbolTable() # Symbol table for dictionary
        # Define input layers
        self.sentences       = None
        self.sentence_length = None
        self.y               = None
        # Super constructor
        super(reLSTM, self).__init__(save_file=save_file, name=name)

    def _mark(self, l, h, idx):
        """Produce markers based on argument positions
            @l: sentence position of first word in argument
            @h: sentence position of last word in argument
            @idx: argument index (0 or 1)
        """
        return [(l, "{}{}".format('[[', idx)), (h+1, "{}{}".format(idx, ']]'))]

    def _mark_sentence(self, s, args):
        """Insert markers around relation arguments in word sequence
            @s: list of tokens in sentence
            @args: list of triples (l, h, idx) as per @_mark(...) corresponding
                   to relation arguments
        Example: Then Barack married Michelle.  
             ->  Then [[1 Barack 1]] married [[2 Michelle 2]].
        """
        marks = sorted([y for m in args for y in self._mark(*m)], reverse=True)
        x = list(s)
        for k, v in marks:
            x.insert(k, v)
        return x

    def _preprocess_data(self, candidates, extend=False):
        """Convert candidate sentences to lookup sequences
            @candidates: candidates to process
            @extend: extend symbol table for tokens (train), or lookup (test)?
        """
        sentences = []
        for c in candidates:
            # Get arguments and lemma sequence
            args = [
                (c[0].get_word_start(), c[0].get_word_end(), 1),
                (c[1].get_word_start(), c[1].get_word_end(), 2)
            ]
            s = self._mark_sentence(
                [w.lower() for w in c.get_parent().__dict__[self.tokens]], args
            )
            # Either extend word table or retrieve from it
            retriever = self.word_dict.get if extend else self.word_dict.lookup
            sentences.append(np.array([retriever(w) for w in s]))
        return sentences

    def _make_tensor(self, x):
        """Construct input tensor with padding"""
        batch_size = len(x)
        tx = np.zeros((self.mx_len, batch_size), dtype=np.int32)
        tlen = np.zeros(batch_size, dtype=np.int32)
        # Pad or trim each x
        # TODO: fix for arguments outside max sentence length
        #       ex: if @mx_len==10 and a relation argument is at position 20,
        #           we should try to include the second argument information
        for k, u in enumerate(x):
            lu = min(len(u), self.mx_len)
            tx[0:lu, k] = u[0:lu]
            tx[lu:, k] = 0
            tlen[k] = lu
        return tx, tlen

    def _build(self):
        """Get feed forward step, loss function, and optimizer for LSTM"""
        # Define input layers
        self.sentences = tf.placeholder(tf.int32, [None, None])
        self.sentence_length = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.float32, [None])
        # Embedding layer
        embedding = tf.Variable(
            tf.random_normal((self.n_v, self.dim), mean=0, stddev=1.0),
            name='embedding'
        )
        inputs = tf.nn.embedding_lookup(embedding, self.sentences)
        # Get RNN graph
        batch_size = tf.shape(self.sentences)[1]
        with tf.variable_scope("LSTM") as scope:
            # LSTM architecture
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.dim, state_is_tuple=True)
            # Set RNN
            initial_state = cell.zero_state(batch_size, tf.float32)
            rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell, inputs, initial_state=initial_state, time_major=True
            )
            # Get LSTM variables
            self.save_dict = {
                x.name: x for x in tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            }
        # Embed sentence by computing word means
        summary_vector = tf.transpose(
            tf.transpose(tf.reduce_sum(rnn_outputs, 0), (1,0)) / 
            tf.cast(self.sentence_length, tf.float32), (1,0)
        )
        # Dropout regularization
        if self.dropout is not None:
            summary_vector = tf.nn.dropout(summary_vector, self.dropout)
        # Sigmoid over embedding layer
        W = tf.Variable(tf.random_normal((self.dim, 1), mean=0, stddev=0.01))
        b = tf.Variable(tf.random_normal([1], mean=0, stddev=0.01))
        h = tf.add(tf.reshape(tf.matmul(summary_vector, W), [-1]), b)
        # Unroll [0, 1] marginals
        unrolled_marginals = tf.reshape(self.y, [-1])
        # Positive class marginal
        self.prediction = tf.nn.sigmoid(h) 
        # Set log loss function
        self.loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(h, unrolled_marginals)
        )
        # Backprop trainer
        self.train_fn = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        # Populate linear layer and input variables
        self.save_dict['W'] = W
        self.save_dict['b'] = b
        self.save_dict['embedding'] = embedding

    def train(self, candidates, training_marginals, n_epochs=25, lr=0.01,
        dim=20, batch_size=100, rebalance=False, dropout_rate=None,
        tokens='lemmas', max_sentence_length=None, print_freq=5):
        """Train LSTM model
            @candidates: candidate objects to train on
            @training_marginals: array of marginals for candidates
            @n_epochs: number of training epochs
            @lr: learning rate
            @dim: embedding dimension
            @batch_size: batch size for mini-batch SGD
            @rebalance: rebalance training examples?
            @dropout_rate: rate for tensorflow.nn.dropout(...)
            @max_sentence_length: maximum sentence length for candidates
            @print_freq: number of epochs after which to print status
            @model_name: name of model for logging and saving
        """
        verbose = print_freq > 0
        if verbose:
            print("[{0}] Dimension={1}  LR={2}".format(self.name, dim, lr))
            print("[{0}] Begin preprocessing".format(self.name))
            st = time()
        # Text preprocessing
        self.tokens = tokens
        x_train = self._preprocess_data(candidates, extend=True)
        # Build model
        self.dim = dim
        self.lr = lr
        self.dropout = tf.constant(dropout_rate) if dropout_rate else None
        self.n_v = self.word_dict.s + 1
        self._build()
        # Get training indices
        train_idxs = get_train_idxs(training_marginals, rebalance=rebalance)
        x_train = [x_train[j] for j in train_idxs]
        y_train = np.ravel(training_marginals)[train_idxs]
        # Get max sentence size
        self.mx_len = max(len(x) for x in x_train)
        self.mx_len = int(min(self.mx_len, max_sentence_length or float('Inf')))
        # Run mini-batch SGD
        batch_size = min(batch_size, len(x_train))
        n = len(x_train)
        self.session = tf.Session()
        if verbose:
            print("[{0}] Preprocessing done ({1:.2f}s)".format(
                self.name, time() - st
            ))
            st = time()
            print("[{0}] Begin training  Epochs={1}  Batch={2}".format(
                self.name, n_epochs, batch_size
            ))
        self.session.run(tf.global_variables_initializer())
        for t in range(n_epochs):
            epoch_loss = 0.0
            for i in range(0, len(x_train), batch_size):
                # Get batch tensors
                r = min(n-1, i+batch_size)
                y_batch = y_train[i:r]
                x_batch, x_batch_lens = self._make_tensor(x_train[i:r])
                # Run training step and evaluate loss function                  
                epoch_loss += self.session.run([self.loss, self.train_fn], {
                    self.sentences: x_batch,
                    self.sentence_length: x_batch_lens,
                    self.y: y_batch,
                })[0]
            # Print training stats
            if verbose and (t % print_freq == 0 or t in [0, (n_epochs-1)]):
                print("[{0}] Epoch {1} ({2:.2f}s)\tAverage loss={3:.6f}".format(
                    self.name, t, time() - st, epoch_loss / n
                ))
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))

    def marginals(self, test_candidates):
        """Feed forward step for marginals"""
        x_test = self._preprocess_data(test_candidates, extend=False)
        x, x_lens = self._make_tensor(x_test)
        return np.ravel(self.session.run([self.prediction], {
            self.sentences: x,
            self.sentence_length: x_lens,
        }))

    def save_info(self, model_name):
        z = (self.mx_len, self.word_dict, self.dim,
             self.tokens, self.n_v, self.lr, self.dropout)
        with open('{0}.info'.format(model_name), 'wb') as f:
            cPickle.dump(z, f)

    def load_info(self, model_name):
        with open('{0}.info'.format(model_name), 'rb') as f:
            (self.mx_len, self.word_dict, self.dim, 
             self.tokens, self.n_v, self.lr, self.dropout) = cPickle.load(f)


class SymbolTable:
    """Wrapper for dict to encode unknown symbols"""
    def __init__(self, starting_symbol=2): 
        self.s = starting_symbol
        self.d = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w):
        return self.d.get(w, 1)
