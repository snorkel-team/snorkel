from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from future.utils import iteritems

import numpy as np
import tensorflow as tf


class SymbolTable(object):
    """Wrapper for dict to encode unknown symbols"""
    def __init__(self, starting_symbol=2, unknown_symbol=1): 
        self.s       = starting_symbol
        self.unknown = unknown_symbol
        self.d       = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w):
        return self.d.get(w, self.unknown)

    def lookup_strict(self, w):
        return self.d.get(w)

    def len(self):
        return self.s

    def reverse(self):
        return {v: k for k, v in iteritems(self.d)}


def scrub(s):
    return ''.join(c for c in s if ord(c) < 128)


def candidate_to_tokens(candidate, token_type='words'):
    tokens = candidate.get_parent().__dict__[token_type]
    return [scrub(w).lower() for w in tokens]


def get_rnn_output(output, dim, lengths):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    index      = tf.range(0, batch_size) * max_length + (lengths - 1)
    flat       = tf.reshape(output, [-1, dim])
    return tf.gather(flat, index)


def get_bi_rnn_output(output, dim, lengths):
    c_output   = tf.concat(output, 2)
    batch_size = tf.shape(c_output)[0]
    max_length = tf.shape(c_output)[1]
    index      = tf.range(0, batch_size) * max_length + (lengths - 1)
    flat       = tf.reshape(c_output, [-1, 2 * dim])
    return tf.gather(flat, index)
