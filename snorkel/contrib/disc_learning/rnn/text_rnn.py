import numpy as np

from rnn_base import RNNBase
from utils import SymbolTable


class TextRNN(RNNBase):
    """TextRNN for strings of text."""
    def _preprocess_data(self, candidates, extend, word_dict=SymbolTable()):
        """Convert candidate sentences to lookup sequences
            @candidates: candidates to process
            @extend: extend symbol table for tokens (train), or lookup (test)?
        """
        data, ends = [], []
        for candidate in candidates:
            toks = candidate.get_contexts()[0].text.split()
            # Either extend word table or retrieve from it
            f = word_dict.get if extend else word_dict.lookup
            data.append(np.array(map(f, toks)))
            ends.append(len(toks))
        return data, ends, word_dict
