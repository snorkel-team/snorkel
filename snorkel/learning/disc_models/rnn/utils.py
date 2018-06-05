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


def get_rnn_output(output, dim, lengths, bi=False, pooling='last'):
    """Take the RNN outputs (state vectors) and pool them according to one of
    the following strategies:
    - pooling=last: Take the last vector from each sequence
    - pooling=mean: Take the mean of the state vectors
    - pooling=max: Take the max of the state vectors
    """
    # Handle bi-directional
    if bi:
        c_output = tf.concat(output, 2)
        d = 2 * dim
    else:
        c_output = output
        d = dim

    batch_size = tf.shape(c_output)[0]
    max_length = tf.shape(c_output)[1]
    
    # Take the last state vector
    if pooling == 'last':
        index = tf.range(0, batch_size) * max_length + (lengths - 1)
        flat  = tf.reshape(c_output, [-1, d])
        return tf.gather(flat, index)
    # Take the max of the vectors
    elif pooling == 'max':
        return tf.reduce_max(c_output, axis=1)
    # Take the mean of the vectors
    elif pooling == 'mean':
        # TODO
        raise Exception("TODO: Mean pooling not implemented yet.")
    else:
        raise Exception("Pooling argument %s not recognized" % (pooling,))
