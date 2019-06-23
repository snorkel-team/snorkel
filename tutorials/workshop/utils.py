import itertools
import torch
import torch.nn as nn

from collections import Counter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.nn.utils.rnn as rnn_utils
from torchtext.vocab import Vocab

class Encoder(nn.Module):
    """The Encoder implements the encode() method, which maps a batch of data to
    encoded output of dimension [batch_size, max_seq_len, encoded_size]

    The first argument must be the encoded size of the Encoder output.

    Args:
        encoded_size: (int) Output feature dimension of the Encoder
    """

    def __init__(self, encoded_size, verbose=True):
        super().__init__()
        self.encoded_size = encoded_size

    def encode(self, X):
        """
        Args:
            X: (torch.LongTensor) of shape [batch_size, max_seq_length,
            encoded_size], with all-0s vectors as padding.
        """
        assert X.shape[-1] == self.encoded_size
        return X.float()


class EmbeddingsEncoder(Encoder):
    def __init__(
        self,
        encoded_size,
        vocab_size=None,
        embeddings=None,
        freeze=False,
        verbose=True,
        seed=123,
        **kwargs,
    ):
        """
        Args:
            encoded_size: (in) Output feature dimension of the Encoder, and
                input feature dimension of the LSTM
            vocab_size: The size of the vocabulary of the embeddings
                If embeddings=None, this helps to set the size of the randomly
                    initialized embeddings
                If embeddings != None, this is used to double check that the
                    provided embeddings have the intended size
            embeddings: An optional embedding Tensor
            freeze: If False, allow the embeddings to be updated
        """
        super().__init__(encoded_size)
        self.verbose = verbose

        # Load provided embeddings or randomly initialize new ones
        if embeddings is None:
            # Note: Need to set seed here for deterministic init
            if seed is not None:
                self._set_seed(seed)
            self.embeddings = nn.Embedding(vocab_size, encoded_size)
            if self.verbose:
                print(f"Using randomly initialized embeddings.")
        else:
            self.embeddings = self._load_pretrained(embeddings)
            if self.verbose:
                print(f"Using pretrained embeddings.")

        # Freeze or not
        self.embeddings.weight.requires_grad = not freeze

        if self.verbose:
            print(
                f"Embeddings shape = ({self.embeddings.num_embeddings}, "
                f"{self.embeddings.embedding_dim})"
            )
            print(f"The embeddings are {'' if freeze else 'NOT '}FROZEN")

    def _set_seed(self, seed):
        self.seed = seed
        if torch.cuda.is_available():
            # TODO: confirm this works for gpus without knowing gpu_id
            # torch.cuda.set_device(self.config['gpu_id'])
            torch.backends.cudnn.enabled = True
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

    def _load_pretrained(self, pretrained):
        if not pretrained.dim() == 2:
            msg = (
                f"Provided embeddings have shape {pretrained.shape}. "
                "Expected a 2-dimensional tensor."
            )
            raise ValueError(msg)
        rows, cols = pretrained.shape
        embedding = nn.Embedding(num_embeddings=rows, embedding_dim=cols)
        embedding.weight.data.copy_(pretrained)
        return embedding

    def encode(self, X):
        """
        Args:
            X: (torch.LongTensor) of shape [batch_size, max_seq_length],
            containing the indices of the embeddings to look up for each item in
            the batch, or 0 for padding.
        """
        return self.embeddings(X.long())


                  
class Featurizer(object):
    def fit(self, input):
        """
        Args:
            input: An iterable of raw data of the appropriate type to be
                featurized, where input[i] corresponds to item i.
        """
        raise NotImplementedError

    def transform(self, input):
        """
        Args:
            input: An iterable of raw data of the appropriate type to be
                featurized, where input[i] corresponds to item i.
        Returns:
            X: A Tensor of features of shape (num_items, ...)
        """
        raise NotImplementedError

    def fit_transform(self, input):
        """Execute fit and transform in sequence."""
        self.fit(input)
        X = self.transform(input)
        return X

class EmbeddingFeaturizer(Featurizer):
    """Converts lists of tokens into a padded Tensor of embedding indices."""

    def __init__(self, markers=[]):
        self.specials = markers + ["<pad>"]
        self.vocab = None

    def build_vocab(self, counter):
        return Vocab(counter, specials=self.specials)

    def fit(self, sents):
        """Builds a vocabulary object based on the tokens in the input.

        Args:
            sents: A list of lists of tokens (representing sentences)

        Vocab kwargs include:
            max_size
            min_freq
            specials
            unk_init
        """
        tokens = list(itertools.chain.from_iterable(sents))
        counter = Counter(tokens)
        self.vocab = self.build_vocab(counter)

    def transform(self, sents):
        """Converts lists of tokens into a Tensor of embedding indices.

        Args:
            sents: A list of lists of tokens (representing sentences)
                NOTE: These sentences should already be marked using the
                mark_entities() helper.
        Returns:
            X: A Tensor of shape (num_items, max_seq_len)
        """

        def convert(tokens):
            return torch.tensor([self.vocab.stoi[t] for t in tokens], dtype=torch.long)

        if self.vocab is None:
            raise Exception(
                "Must run .fit() for .fit_transform() before " "calling .transform()."
            )

        seqs = sorted([convert(s) for s in sents], key=lambda x: -len(x))
        X = torch.LongTensor(pad_sequence(seqs, batch_first=True))
        return X


def mark_entities(tokens, positions, markers=[], style="insert"):
    """Adds special markers around tokens at specific positions (e.g., entities)

    Args:
        tokens: A list of tokens (the sentence)
        positions:
            1) A list of inclusive ranges (tuples) corresponding to the
            token ranges of the entities in order. (Assumes each entity
            has only one corresponding mention.)
            OR
            2) A dict of lists with keys corresponding to mention indices and
            values corresponding to one or more inclusive ranges corresponding
            to that mention. (Allows entities to potentially have multiple
            mentions)
        markers: A list of strings (length of 2 * the number of entities) to
            use as markers of the entities.
        style: Where to apply the markers:
            'insert': Insert the markers as new tokens before/after each entity
            'concatenate': Prepend/append the markers to the first/last token
                of each entity
            If the tokens are going to be input to an LSTM, then it is usually
            best to use the 'insert' option; 'concatenate' may be better for
            viewing.

    Returns:
        toks: An extended list of tokens with markers around the mentions

    WARNING: if the marked token set will be used with pretrained embeddings,
        provide markers that will not result in UNK embeddings!

    Example:
        Input:  (['The', 'cat', 'sat'], [(1,1)])
        Output: ['The', '[[BEGIN0]]', 'cat', '[[END0]]', 'sat']
    """
    if markers and len(markers) != 2 * len(positions):
        msg = (
            f"Expected len(markers) == 2 * len(positions), "
            f"but {len(markers)} != {2 * len(positions)}."
        )
        raise ValueError(msg)

    toks = list(tokens)

    # markings will be of the form:
    # [(position, entity_idx), (position, entity_idx), ...]
    if isinstance(positions, list):
        markings = [(position, idx) for idx, position in enumerate(positions)]
    elif isinstance(positions, dict):
        markings = []
        for idx, v in positions.items():
            for position in v:
                markings.append((position, idx))
    else:
        msg = (
            f"Argument _positions_ must be a list or dict. "
            f"Instead, got {type(positions)}"
        )
        raise ValueError(msg)

    markings = sorted(markings)
    for i, ((si, ei), idx) in enumerate(markings):
        if markers:
            start_marker = markers[2 * idx]
            end_marker = markers[2 * idx + 1]
        else:
            start_marker = f"[[BEGIN{idx}]]"
            end_marker = f"[[END{idx}]]"
        if style == "insert":
            toks.insert(si + 2 * i, start_marker)
            toks.insert(ei + 2 * (i + 1), end_marker)
        elif style == "concatenate":
            toks[si] = start_marker + toks[si]
            toks[ei] = toks[ei] + end_marker
        else:
            raise NotImplementedError
    return toks