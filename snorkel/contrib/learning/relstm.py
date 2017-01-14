import cPickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from snorkel.learning import NoiseAwareModel
from tensorflow.python.ops.functional_ops import map_fn
from time import asctime, time


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


def time_str():
    return asctime().replace(' ', '-').replace(':', '-')


class reLSTM(NoiseAwareModel):

    def __init__(self, save_file=None, **kwargs):
        """LSTM for relation extraction"""
        self.mx_len     = None
        self.prediction = None
        self.session    = None
        self.word_dict  = SymbolTable()
        # Define input layers
        self.sentences = tf.placeholder(tf.int32, [None, None])
        self.sentence_length = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.float32, [None])
        # Load model
        if save_file is not None:
            self.load(save_file)
        # Super constructor
        super(reLSTM, self).__init__(**kwargs)

    def _mark(self, l, h, idx):
        """Produce markers based on argument positions"""
        return [(l, "{}{}".format('[[', idx)), (h+1, "{}{}".format(idx, ']]'))]

    def _mark_sentence(self, s, mids):
        """Insert markers around relation arguments in word sequence
        Example: Then Barack married Michelle.  
             ->  Then [[0 Barack 0]] married [[1 Michelle 1]].
        """
        marks = sorted([y for m in mids for y in self._mark(*m)], reverse=True)
        x = list(s)
        for k, v in marks:
            x.insert(k, v)
        return x

    def _preprocess_data(self, candidates, extend=False):
        sentences = []
        for c in candidates:
            # Get arguments and lemma sequence
            args = [
                (c[0].get_word_start(), c[0].get_word_end(), 1),
                (c[1].get_word_start(), c[1].get_word_end(), 2)
            ]
            s = self._mark_sentence([w.lower() for w in c.get_parent().lemmas], args)
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
        # TODO: fix for arguments outside max length
        for k, u in enumerate(x):
            lu = min(len(u), self.mx_len)
            tx[0:lu, k] = u[0:lu]
            tx[lu:, k] = 0
            tlen[k] = lu
        return tx, tlen

    def _build_lstm(self, sents, sent_lens, marginals, lr, dim, dropout, n_v):
        """Get feed forward step, cost function, and optimizer for LSTM"""
        # Get simple architecture
        cell = tf.nn.rnn_cell.BasicLSTMCell(dim, state_is_tuple=True)
        batch_size = tf.shape(sents)[1]
        # Set input layers
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", (n_v, dim), dtype=tf.float32
            )
            inputs = tf.nn.embedding_lookup(embedding, sents)
        # Set RNN
        initial_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, _ = tf.nn.dynamic_rnn(
            cell, inputs, initial_state=initial_state, time_major=True
        )
        # Take mean across sentences
        summary_vector = tf.transpose(
            tf.transpose(tf.reduce_sum(rnn_outputs, 0), (1,0)) / 
            tf.cast(sent_lens, tf.float32), (1,0)
        )
        # Dropout regularization
        if dropout is not None:
            summary_vector = tf.nn.dropout(summary_vector, dropout)
        # Sigmoid over embedding layer
        W = tf.Variable(tf.truncated_normal((dim, 1), stddev=1e-2))
        b = tf.Variable(tf.truncated_normal([1], stddev=1e-2))
        u = tf.reshape(tf.matmul(summary_vector, W), [-1])
        # Unroll [0, 1] marginals
        unrolled_marginals = tf.reshape(marginals, [-1])            
        # Positive class marginal
        prediction = tf.nn.sigmoid(u + b) 
        # Set log loss cost function
        cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(u + b, unrolled_marginals)
        )
        # Backprop trainer
        train_fn  = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)
        return prediction, cost, train_fn

    def train(self, candidates, marginals, n_epochs=25, lr=0.01, dim=20,
        batch_size=100, rebalance=False, dropout_rate=None,
        max_sentence_length=None, n_print=5, model_name=None):
        """ Train LSTM model """
        # Check input sizes
        if len(candidates) != len(marginals):
            raise Exception("{0} candidates and {1} marginals".format(
                len(candidates), len(marginals))
            )
        # Starter message
        verbose = n_print > 0
        if verbose:
            print("[reLSTM]  Dimension={}  LR={}".format(dim, lr))
            print("[reLSTM] Begin preprocessing")
            st = time()
        # Text preprocessing
        train_x = self._preprocess_data(candidates, extend=True)
        # Build model
        dropout = None if dropout_rate is None else tf.constant(dropout_rate)
        self.prediction, cost, train_fn = self._build_lstm(
            self.sentences, self.sentence_length, self.y, lr, dim,
            dropout, self.word_dict.s + 1
        )
        # Get training counts
        if rebalance:
            pos, neg = np.where(marginals > 0.5)[0], np.where(marginals < 0.5)[0]
            k = min(len(pos), len(neg))
            idxs = np.concatenate((
                np.random.choice(pos, size=k, replace=False),
                np.random.choice(neg, size=k, replace=False)
            ))
        else:
            idxs = np.ravel(np.where(np.abs(marginals - 0.5) > 1e-6)[0])
        # Shuffle training data
        np.random.shuffle(idxs)
        train_x, y = [train_x[j] for j in idxs], marginals[idxs]
        # Get max sentence size
        self.mx_len = max(len(x) for x in train_x)
        self.mx_len = int(min(self.mx_len, max_sentence_length or float('Inf')))
        # Run mini-batch SGD
        batch_size = min(batch_size, len(train_x))
        self.session = tf.Session()
        if verbose:
            print("[reLSTM] Preprocessing done ({0:.2f}s)".format(time()-st))
            st = time()
            print("[reLSTM] Begin training  Epochs={0}  Batch={1}".format(
                n_epochs, batch_size
            ))
        self.session.run(tf.global_variables_initializer())
        for t in range(n_epochs):
            epoch_loss = 0.0
            for i in range(0, len(train_x), batch_size):
                # Get batch tensors
                y_batch = y[i:i+batch_size]
                x_batch, x_batch_lens = self._make_tensor(
                    train_x[i:i+batch_size]
                )
                # Run training step and evaluate cost function                  
                epoch_loss += self.session.run([cost, train_fn], {
                    self.sentences: x_batch,
                    self.sentence_length: x_batch_lens,
                    self.y: y_batch,
                })[0]
            # Print training stats
            if verbose and ((t+1) % n_print == 0 or t == (n_epochs-1)):
                print("[reLSTM] Epoch {0} ({1:.2f}s)\tLoss={2:.6f}".format(
                    t+1, time() - st, epoch_loss
                ))
        # Save model
        if verbose:
            print("[reLSTM] Training done ({0:.2f}s)".format(time() - st))
        self.save(dim, model_name, verbose)

    def marginals(self, test_candidates):
        """Feed forward step for marginals"""
        if any(z is None for z in [self.session, self.prediction]):
            raise Exception("[reLSTM] Model not defined")
        test_x = self._preprocess_data(test_candidates, extend=False)
        # Get input tensors with dummy marginals
        x, x_lens = self._make_tensor(test_x)
        return np.ravel(self.session.run([self.prediction], {
            self.sentences: x,
            self.sentence_length: x_lens,
            self.y: np.empty(len(x))
        }))

    def save(self, dim, model_name=None, verbose=False):
        """Save model"""
        model_name = model_name or ("relstm_" + time_str())
        saver = tf.train.Saver()
        saver.save(self.session, model_name)
        with open("{0}.info".format(model_name), 'wb') as f:
            cPickle.dump((dim, self.mx_len, self.word_dict), f)
        print("[reLSTM] Model saved. To load, use name\n\t\t{0}".format(model_name))

    def load(self, model_name):
        """Load model"""
        with open("{0}.info".format(model_name), 'rb') as f:
            dim, self.mx_len, self.word_dict = cPickle.load(f)
        self.prediction, _, _ = self._build_lstm(
            self.sentences, self.sentence_length, self.y, 0.0, dim,
            None, self.word_dict.s + 1
        )
        saver = tf.train.Saver()
        self.session = tf.Session()
        saver.restore(self.session, '{0}.meta'.model_name)
        print("[reLSTM] Successfully loaded model <{0}>".format(model_name))
