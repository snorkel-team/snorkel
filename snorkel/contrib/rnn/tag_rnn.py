import numpy as np

from rnn_base import RNNBase
from utils import candidate_to_tokens, SymbolTable


def tag(seq, labels):
    assert(len(seq) == len(labels))
    seq_new, t = [], False
    for x, y in zip(seq, labels):
        if y and (not t):
            seq_new.append(self.OPEN)
            seq_new.append(x)
            t = True
        elif (not y) and t:
            seq_new.append(self.CLOSE)
            seq_new.append(x)
            t = False
        else:
            seq_new.append(x)
    return seq_new


class TagRNN(RNNBase):
    """TagRNN for sequence tagging"""
    OPEN, CLOSE = '~~[[~~', '~~]]~~'

    def _preprocess_data(self, candidates, extend, word_dict=SymbolTable()):
        """Convert candidate sentences to tagged symbol sequences
            @candidates: candidates to process
            @extend: extend symbol table for tokens (train), or lookup (test)?
        """
        data, ends = [], []
        for candidate in candidates:
            # Read sentence data
            tokens = candidate_to_tokens(candidate)
            # Get label sequence
            labels = np.zeros(len(tokens), dtype=int)
            labels[c[0].get_word_start() : c[0].get_word_end()+1] = 1
            # Tag sequence
            s = tag(tokens, labels)
            # Either extend word table or retrieve from it
            f = word_dict.get if extend else word_dict.lookup
            data.append(np.array(map(f, s)))
            ends.append(c[0].get_word_end())
        return data, ends, word_dict
