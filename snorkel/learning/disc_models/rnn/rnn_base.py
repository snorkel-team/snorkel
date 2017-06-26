import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import warnings

from snorkel.learning import LabelBalancer, TFNoiseAwareModel
from snorkel.models import Candidate
from utils import get_bi_rnn_output, SymbolTable
from time import time


SD = 0.1


class RNNBase(TFNoiseAwareModel):
    representation = True

    def _preprocess_data(self, candidates, extend, word_dict=SymbolTable()):
        """Build @self.word_dict to encode and process data for extraction
            Return list of encoded sentences, list of last index of arguments,
            and the word dictionary (extended if extend=True)
        """
        raise NotImplementedError()

    def _check_max_sentence_length(self, ends, max_len=None):
        """Check that extraction arguments are within @self.max_len"""
        mx = max_len or self.max_len
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
        x_batch    = np.zeros((batch_size, self.max_len), dtype=np.int32)
        len_batch  = np.zeros(batch_size, dtype=np.int32)
        for j, token_ids in enumerate(x):
            t               = min(len(token_ids), self.max_len)
            x_batch[j, 0:t] = token_ids[0:t]
            len_batch[j]    = t
        return x_batch, len_batch

    def _build_model(self, dim=50, attn_window=None, max_len=20,
        cell_type=rnn.BasicLSTMCell, word_dict=SymbolTable(), **kwargs):
        """
        Build RNN model
        @dim: embedding dimension
        @attn_window: attention window length (no attention if 0 or None)
        @cell_type: subclass of tensorflow.python.ops.rnn_cell_impl._RNNCell
        @batch_size: batch size for mini-batch SGD
        @vocab_size: Vocab size for determining size of word embeddings tensor
        """
        # Set the word dictionary passed in as the word_dict for the instance
        self.max_len = max_len
        self.word_dict = word_dict
        vocab_size = word_dict.len()

        # Define input layers
        self.sentences        = tf.placeholder(tf.int32, [None, None])
        self.sentence_lengths = tf.placeholder(tf.int32, [None])

        # Seeds
        s = self.seed
        s1, s2, s3, s4 = [None] * 4 if s is None else [s+i for i in range(4)]

        # Embedding layer
        emb_var = tf.Variable(
            tf.random_normal((vocab_size - 1, dim), stddev=SD, seed=s1))
        embedding = tf.concat([tf.zeros([1, dim]), emb_var], axis=0)
        inputs = tf.nn.embedding_lookup(embedding, self.sentences)
        
        # Build RNN graph
        batch_size = tf.shape(self.sentences)[0]
        init = tf.contrib.layers.xavier_initializer(seed=s2)
        with tf.variable_scope(self.name, reuse=False, initializer=init):
            # Build RNN cells
            fw_cell = cell_type(dim)
            bw_cell = cell_type(dim)
            # Add attention if needed
            if attn_window:
                fw_cell = rnn.AttentionCellWrapper(
                    fw_cell, attn_window, state_is_tuple=True
                )
                bw_cell = rnn.AttentionCellWrapper(
                    bw_cell, attn_window, state_is_tuple=True
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
        potentials = get_bi_rnn_output(rnn_out, dim, self.sentence_lengths)
        
        # Add dropout layer
        self.keep_prob = tf.placeholder(tf.float32)
        potentials_dropout = tf.nn.dropout(potentials, self.keep_prob, seed=s3)

        # Build activation layer
        if self.cardinality > 2:
            self.Y = tf.placeholder(tf.float32, [None, self.cardinality])
            W = tf.Variable(tf.random_normal((2*dim, self.cardinality), 
                stddev=SD, seed=s4))
            b = tf.Variable(np.zeros(self.cardinality), dtype=tf.float32)
            self.logits = tf.matmul(potentials, W) + b
            self.marginals_op = tf.nn.softmax(self.logits)
        else:
            self.Y = tf.placeholder(tf.float32, [None])
            W = tf.Variable(tf.random_normal((2*dim, 1), stddev=SD, seed=s4))
            b = tf.Variable(0., dtype=tf.float32)
            self.logits = tf.squeeze(tf.matmul(potentials, W)) + b
            self.marginals_op = tf.nn.sigmoid(self.logits)

    def _construct_feed_dict(self, X_b, Y_b, lr=0.01, dropout=None, **kwargs):
        X_b, len_b = self._make_tensor(X_b)
        return {
            self.sentences:        X_b,
            self.sentence_lengths: len_b,
            self.Y:                Y_b,
            self.keep_prob:        dropout or 1.0,
            self.lr:               lr
        }

    def train(self, X_train, Y_train, X_dev=None, max_sentence_length=None, 
        **kwargs):
        """
        Perform preprocessing of data, construct dataset-specific model, then
        train.
        """
        # Text preprocessing
        X_train, ends, word_dict = self._preprocess_data(X_train, extend=True)
        if X_dev is not None:
            X_dev, _, _ = self._preprocess_data(X_dev, word_dict=word_dict, 
                extend=False)
        
        # Get max sentence size
        max_len = max_sentence_length or max(len(x) for x in X_train)
        self._check_max_sentence_length(ends, max_len=max_len)
        
        # Train model
        super(RNNBase, self).train(X_train, Y_train, X_dev=X_dev,
            word_dict=word_dict, max_len=max_len, **kwargs)

    def marginals(self, test_candidates):
        """Get likelihood of tagged sequences represented by test_candidates
            @test_candidates: list of lists representing test sentence
        """
        # Preprocess if not already preprocessed
        if isinstance(test_candidates[0], Candidate):
            X_test, ends, _ = self._preprocess_data(test_candidates, 
                extend=False)
            self._check_max_sentence_length(ends)
        else:
            X_test = test_candidates

        # Make tensor and run prediction op
        x, x_len = self._make_tensor(X_test)
        return self.session.run(self.marginals_op, {
            self.sentences:        x,
            self.sentence_lengths: x_len,
            self.keep_prob:        1.0,
        })
