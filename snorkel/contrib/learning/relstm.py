import cPickle as pkl
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
        self.labels = tf.placeholder(tf.int32, [None])
        # Load model
        if save_file is not None:
            self.load(save_file)
        # Super constructor
        super(reLSTM, self).__init__(**kwargs)

    def _gen_marks(self, l, h, idx):
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
            s = mark_sentence([w.lower() for w in c.get_parent().lemmas], args)
            # Either extend word table or retrieve from it
            retriever = self.word_dict.get if extend else self.word_dict.lookup
            sentences.append(np.array([retriever(w) for w in s]))
        return sentences

    def _make_tensor(self, x, y=None):
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
        return tx, (np.zeros(len(tx)) if y is None else np.ravel(y)), tlen

    def _build_lstm(self, sents, sent_lens, labels, lr, n_hidden, dropout, n_v):
        """Get feed forward step, cost function, and optimizer for LSTM"""
        # Get simple architecture
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
        batch_size = tf.shape(sents)[1]
        # Set input layers
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", (n_v, n_hidden), dtype=tf.float32
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
        W = tf.Variable(tf.truncated_normal((n_hidden, 1), stddev=1e-2))
        b = tf.Variable(tf.truncated_normal([1], stddev=1e-2))
        u = tf.reshape(tf.matmul(summary_vector, W), [-1])
        # Unroll {-1, 1} hard labels
        unrolled_labels = tf.reshape(labels, [-1])            
        # Positive class marginal
        prediction = tf.nn.sigmoid(u + b) 
        # Set log loss cost function
        cost = -tf.reduce_mean(tf.log(tf.nn.sigmoid(
            tf.mul(u + b, tf.cast(unrolled_labels, tf.float32))
        )))
        # Backprop trainer
        train_fn  = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)
        return prediction, cost, train_fn

    def train(self, candidates, y, n_epochs=10, lr=0.01, n_hidden=20,
        batch_size=100, rebalance=False, dropout_rate=None,
        max_sentence_length=None, n_print=50, model_name=None):
        """ Train LSTM model """
        verbose = n_print > 0
        if verbose:
            print("[reLSTM] Layers={} LR={}".format(n_hidden, lr))
            print("[reLSTM] Begin preprocessing")
            st = time()
        # TODO: standardize input labels
        if any(yy not in [1, -1] for yy in y):
            raise Exception("Labels should be {-1, 1}")
        # Text preprocessing
        train_x = self._preprocess_data(candidates)
        # Build model
        dropout = None if dropout_rate is None else tf.constant(dropout_rate)
        self.prediction, cost, train_fn = build_lstm(
            self.sentences, self.sentence_length, self.labels, lr, n_hidden,
            dropout, self.word_dict.current_symbol + 1
        )
        # Get training counts 
        if rebalance:
            pos, neg = np.where(train_y == 1)[0], np.where(train_y == -1)[0]
            k = min(len(pos), len(neg))
            idxs = np.concatenate((
                np.random.choice(pos, size=k, replace=False),
                np.random.choice(neg, size=k, replace=False)
            ))
        else:
            idxs = np.ravel(xrange(len(train_y)))
        # Shuffle training data
        np.random.shuffle(idxs)
        train_x, train_y = [train_x[j] for j in idxs], train_y[idxs]
        # Get max sentence size
        self.mx_len = max(train_x, lambda x: len(x))
        self.mx_len = int(min(self.mx_len, max_sentence_length or float('Inf')))
        # Run mini-batch SGD
        batch_size = min(batch_size, len(train_x))
        self.session = tf.Session()
        if verbose:
            print("[reLSTM] Preprocessing done ({0:.2f}s)".format(time()-st))
            st = time()
            print("[reLSTM] Begin training\tEpochs={0}\tBatch={1}".format(
                n_epochs, batch_size
            ))
        self.session.run(tf.global_variables_initializer())
        for t in range(n_epochs):
            epoch_error = 0
            for i in range(0, len(train_x), batch_size):
                # Get batch tensors
                y_batch = train_y[i:i+batch_size]
                x_batch, x_batch_lens = self._make_tensor(
                    train_x[i:i+batch_size]
                )
                # Run training step and evaluate cost function                  
                epoch_error += self.session.run([cost, train_fn], {
                    self.sentences: x_batch,
                    self.sentence_length: x_batch_lens,
                    self.labels: y_batch,
                })[0]
            # Print training stats
            if verbose and (t % n_print == 0 or t == (n_epochs - 1)):
                print("[reLSTM] Epoch {0} ({1:.2f}s)\tError={2:.6f}".format(
                    t, time.time() - st, epoch_error
                ))
        # Save model
        self.save(model_name)        
        if verbose:
            print("[reLSTM] Training done ({0:.2f}s)".format(time.time() - st))
            print("[reLSTM] Model saved in file: {}".format(model_name))

    def marginals(self, test_candidates):
        """Feed forward step for marginals"""
        if any(z is None for z in [self.session, self.prediction]):
            raise Exception("[reLSTM] Model not defined")
        test_x = self._preprocess_data(test_candidates)
        # Get input tensors with dummy labels
        x, y, x_lens = self._make_tensor(test_x)
        return np.ravel(self.session.run([self.prediction], {
            self.sentences: x,
            self.sentence_length: x_lens,
            self.labels: y
        }))

    def save(self, model_name=None):
        """Save model"""
        model_name = model_name or ("relstm_" + time_str())
        saver = tf.train.Saver()
        saver.save(self.session, "./{0}.session".format(model_name))
        with open("./{0}.info".format(model_name)) as f:
            # TODO: save prediction, mx_len
            pass

    def load(self, model_name):
        """Load model"""
        # TODO: load info and session file
        pass
