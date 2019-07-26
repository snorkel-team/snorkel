from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import numpy as np
import torch

from snorkel.learning.pytorch import TorchNoiseAwareModel
from snorkel.models import Candidate
from snorkel.learning.pytorch.rnn.utils import candidate_to_tokens, SymbolTable


def mark(l, h, idx):
    """Produce markers based on argument positions
    
    :param l: sentence position of first word in argument
    :param h: sentence position of last word in argument
    :param idx: argument index (1 or 2)
    """
    return [(l, "{}{}".format('~~[[', idx)), (h+1, "{}{}".format(idx, ']]~~'))]


def mark_sentence(s, args):
    """Insert markers around relation arguments in word sequence
    
    :param s: list of tokens in sentence
    :param args: list of triples (l, h, idx) as per @_mark(...) corresponding
               to relation arguments
    
    Example: Then Barack married Michelle.  
         ->  Then ~~[[1 Barack 1]]~~ married ~~[[2 Michelle 2]]~~.
    """
    marks = sorted([y for m in args for y in mark(*m)], reverse=True)
    x = list(s)
    for k, v in marks:
        x.insert(k, v)
    return x


class RNNBase(TorchNoiseAwareModel):
    representation = True
    
    def initialize_hidden_state(self, batch_size):
        raise NotImplementedError
    
    def _pytorch_outputs(self, X, batch_size):
        n = len(X)
        if not batch_size:
            batch_size = len(X)
        
        if isinstance(X[0], Candidate):
            X = self._preprocess_data(X, extend=False)
        
        outputs = torch.Tensor([])
        
        for batch in range(0, n, batch_size):
            
            if batch_size > len(X[batch:batch+batch_size]):
                batch_size = len(X[batch:batch+batch_size])
    
            hidden_state = self.initialize_hidden_state(batch_size)
            max_batch_length = max(map(len, X[batch:batch+batch_size]))
            
            padded_X = torch.zeros((batch_size, max_batch_length), dtype=torch.long)
            for idx, seq in enumerate(X[batch:batch+batch_size]):
                # TODO: Don't instantiate tensor for each row
                padded_X[idx, :len(seq)] = torch.LongTensor(seq)

            output = self.forward(padded_X, hidden_state)

            # TODO: Does skipping the cat when there is only one batch speed things up significantly?
            if self.cardinality == 2:
                outputs = torch.cat((outputs, output.view(-1)), 0)
            else:
                outputs = torch.cat((outputs, output), 0)

        return outputs

    def _preprocess_data(self, candidates, extend=False):
        """Convert candidate sentences to lookup sequences
        
        :param candidates: candidates to process
        :param extend: extend symbol table for tokens (train), or lookup (test)?
        """
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()
        data = []
        for candidate in candidates:
            # Mark sentence
            args = [(candidate[i].get_word_start(), candidate[i].get_word_end(), i+1) for i in range(len(candidate))]
            s = mark_sentence(candidate_to_tokens(candidate), args)
            # Either extend word table or retrieve from it
            f = self.word_dict.get if extend else self.word_dict.lookup
            data.append(np.array(list(map(f, s))))
            
        return data

    def train(self, X_train, Y_train, X_dev=None, **kwargs):
        # Preprocesses data, including constructing dataset-specific dictionary
        X_train = self._preprocess_data(X_train, extend=True)
        if X_dev is not None:
            X_dev = self._preprocess_data(X_dev, extend=False)

        # Note we pass word_dict through here so it gets saved...
        super(RNNBase, self).train(X_train, Y_train, X_dev=X_dev,
            word_dict=self.word_dict, **kwargs)

