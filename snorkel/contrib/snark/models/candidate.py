import pickle

from snorkel.models import Sentence, Span
from . import SparkModel


class Candidate(SparkModel):
    """An abstract candidate relation."""
    def __init__(self, id, split, context_names, contexts, cids, name='Candidate'):
        self.id = id
        self.split = split
        self.name = name
        self.__argnames__ = context_names
        for i, name in enumerate(context_names):
            setattr(self, name, contexts[i])
            setattr(self, name + '_cid', cids[i])

    def get_contexts(self):
        """Get a tuple of the consituent contexts making up this candidate"""
        return tuple(getattr(self, name) for name in self.__argnames__)

    def get_parent(self):
        # Fails if both contexts don't have same parent
        p = [c.get_parent() for c in self.get_contexts()]
        if p.count(p[0]) == len(p):
            return p[0]
        else:
            raise Exception("Contexts do not all have same parent")

    def get_cids(self):
        """Get a tuple of the canonical IDs (CIDs) of the contexts making up
        this candidate.
        """
        return tuple(getattr(self, name + "_cid") for name in self.__argnames__)

    def __len__(self):
        return len(self.__argnames__)

    def __getitem__(self, key):
        return self.get_contexts()[key]

    def __repr__(self):
        return "%s(%s)" % (self.name, ", ".join(map(str, self.get_contexts())))


# Note: This should ideally not be hard-coded...
CONTEXT_OFFSET = 2
# Note: We store the sentence here, not the sentence_id
SPAN_COLS = ['id', 'sentence_id', 'char_start', 'char_end', 'meta']
SENTENCE_COLS = ['id', 'document_id', 'position', 'text', 'words',
    'char_offsets', 'abs_char_offsets', 'lemmas', 'pos_tags', 'ner_tags', 'dep_parents',
    'dep_labels', 'entity_cids', 'entity_types']


def wrap_candidate(row, class_name='Candidate', argnames=None):
    """
    Wraps raw tuple from <candidate_classname>_serialized table with object data
    structure

    :param row: raw tuple
    :return: candidate object
    """
    # Infer arity from size of row
    arity = float(len(row) - 2 - len(SENTENCE_COLS)) / (len(SPAN_COLS) + 1)
    assert int(arity) == arity, "%d is not a multiple of %d" % (len(row) - 2 - len(SENTENCE_COLS),
                                                                len(SPAN_COLS) + 1)
    arity = int(arity)

    # NB: We hardcode in an assumed Context hierarchy here:
    # Sentence -> (Spans)
    # Should make more general. Also note that we assume only the local context
    # subtree (Sentence + k Spans comprising the candidate) are provided.
    # Order of columns is id | split | span1.cid | span1.* | ... | sent.*

    # Create Sentence object
    # Arrays are stored as BLOBs so need to be converted to Python using Pickle
    sentence_args = dict(zip(SENTENCE_COLS, row[-len(SENTENCE_COLS):]))
    sentence_args = {k: pickle.loads(v) if isinstance(v, bytearray) else v
                     for k, v in sentence_args.iteritems()}
    sent = Sentence(**sentence_args)

    # Create the Span objects
    spans, cids = [], []
    for i in range(arity):
        j = CONTEXT_OFFSET + i * (len(SPAN_COLS) + 1)
        span_args = dict(zip(SPAN_COLS, row[j+1:j+len(SPAN_COLS)]))
        span_args = {k: pickle.loads(v) if isinstance(v, bytearray) else v
                for k, v in span_args.iteritems()}
        span = Span(**span_args)
        # Store the Sentence in the Span
        span.sentence = sent
        spans.append(span)
        # Get the CID as well
        cids.append(row[j])

    # Create candidate object
    candidate = Candidate(
        id=row[0],
        split=row[1],
        context_names=argnames,
        contexts=spans,
        cids=cids,
        name=class_name
    )

    return candidate
