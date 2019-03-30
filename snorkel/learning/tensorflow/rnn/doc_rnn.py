from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import numpy as np

from .rnn_base import RNNBase
from .utils import SymbolTable


class DocRNN(RNNBase):
    """
    RNN class to process complete documents without breaking them down
    into sentences or spans.
    """
    def _preprocess_data(self, candidates, extend=False):
        """
        Converts candidate documents into lookup sequences of integers.
        
        For each document in the list of candidates, this method breaks
        it into sequence of tokens, and then creates a corresponding 
        sequence of integers based on the lookup dictionary. As it reads
        newer tokens, this method also expands the lookup dictionary if 
        the argument `extend` is True.
        
        :param candidates: List of candidates (i.e. list of TemporaryDocument 
                           objects) to be processed
        :param extend: extend symbol table for tokens (train), or just
                       lookup(test)
        """
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()
        
        data, ends = [], []
        f = self.word_dict.get if extend else self.word_dict.lookup
        for c in candidates:
            doc = c.get_contexts()[0]
            textToks = list()
            for s in doc.sentences:
                textToks.extend(s.text.split())
            data.append(np.array(list(map(f, textToks))))
            ends.append(len(textToks))
        return data, ends
