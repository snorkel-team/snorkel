from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from . import SparkModel
from snorkel.models import TemporarySpan


class Context(SparkModel):
    def get_parent(self):
        raise NotImplementedError()

    def get_children(self):
        raise NotImplementedError()

    def get_sentence_generator(self):
        raise NotImplementedError()


class Document(Context):
    """
    A root Context.
    """
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', None)
        self.name = kwargs.get('name', None)
        self.sentences = kwargs.get('sentences', [])
        self.meta = kwargs.get('meta', {})

    def get_parent(self):
        return None

    def get_children(self):
        return self.sentences

    def get_sentence_generator(self):
        for sentence in self.sentences:
            yield sentence

    def __repr__(self):
        return "Document " + str(self.name)


class Sentence(Context):
    """
    A sentence Context in a Document.
    """
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', None)
        self.document = kwargs.get('document', None)
        self.position = kwargs.get('position', None)
        self.text = kwargs.get('text', "")
        self.words = kwargs.get('words', [])
        self.char_offsets = kwargs.get('char_offsets', [])
        self.lemmas = kwargs.get('lemmas', [])
        self.pos_tags = kwargs.get('pos_tags', [])
        self.ner_tags = kwargs.get('ner_tags', [])
        self.dep_parents = kwargs.get('dep_parents', [])
        self.dep_labels = kwargs.get('dep_labels', [])
        self.entity_cids = kwargs.get('entity_cids', [])
        self.entity_types = kwargs.get('entity_types', [])

    def get_parent(self):
        return self.document

    def get_children(self):
        return None

    def get_sentence_generator(self):
        yield self

    def __repr__(self):
        return "Sentence(%s,%s,%s)" % (self.document, self.position,
            self.text.encode('utf-8'))


class Span(Context, TemporarySpan):
    """
    A span of characters, identified by Context id and character-index start,
    end (inclusive).

    char_offsets are **relative to the Context start**
    """
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', None)
        self.sentence = kwargs.get('sentence', None)
        self.char_start = kwargs.get('char_start', -1)
        self.char_end = kwargs.get('char_end', -1)
        self.meta = kwargs.get('meta', {})

    def get_parent(self):
        return self.sentence

    def get_children(self):
        return None
