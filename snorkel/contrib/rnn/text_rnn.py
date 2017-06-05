import numpy as np

from rnn_base import RNNBase


class TextRNN(RNNBase):

    def __init__(self, save_file=None, name='textRNN', seed=None, n_threads=4):
        """reRNN for relation extraction"""
        super(TextRNN, self).__init__(
            n_threads=n_threads, save_file=save_file, name=name, seed=seed
        )

    def _preprocess_data(self, candidates, extend):
        """Convert candidate sentences to lookup sequences
            @candidates: candidates to process
            @extend: extend symbol table for tokens (train), or lookup (test)?
        """
        data, ends = [], []
        for candidate in candidates:
            toks = candidate.get_contexts()[0].text.split()
            # Either extend word table or retrieve from it
            f = self.word_dict.get if extend else self.word_dict.lookup
            data.append(np.array(map(f, toks)))
            ends.append(len(toks))
        return data, ends
