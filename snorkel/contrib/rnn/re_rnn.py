import numpy as np

from rnn_base import RNNBase
from utils import candidate_to_tokens, SymbolTable


def mark(l, h, idx):
    """Produce markers based on argument positions
        @l: sentence position of first word in argument
        @h: sentence position of last word in argument
        @idx: argument index (1 or 2)
    """
    return [(l, "{}{}".format('~~[[', idx)), (h+1, "{}{}".format(idx, ']]~~'))]


def mark_sentence(s, args):
    """Insert markers around relation arguments in word sequence
        @s: list of tokens in sentence
        @args: list of triples (l, h, idx) as per @_mark(...) corresponding
               to relation arguments
    Example: Then Barack married Michelle.  
         ->  Then ~~[[1 Barack 1]]~~ married ~~[[2 Michelle 2]]~~.
    """
    marks = sorted([y for m in args for y in mark(*m)], reverse=True)
    x = list(s)
    for k, v in marks:
        x.insert(k, v)
    return x


class reRNN(RNNBase):
    """reRNN for relation extraction"""
    def _preprocess_data(self, candidates, extend, word_dict=SymbolTable()):
        """Convert candidate sentences to lookup sequences
            @candidates: candidates to process
            @extend: extend symbol table for tokens (train), or lookup (test)?
        """
        data, ends = [], []
        for candidate in candidates:
            # Mark sentence
            args = [
                (candidate[0].get_word_start(), candidate[0].get_word_end(), 1),
                (candidate[1].get_word_start(), candidate[1].get_word_end(), 2)
            ]
            s = mark_sentence(candidate_to_tokens(candidate), args)
            # Either extend word table or retrieve from it
            f = word_dict.get if extend else word_dict.lookup
            data.append(np.array(map(f, s)))
            ends.append(max(candidate[i].get_word_end() for i in [0, 1]))
        return data, ends, word_dict
