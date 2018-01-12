from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from collections import defaultdict
from snorkel.models import construct_stable_id
from snorkel.parser import Parser, ParserConnection


class SpaCy(Parser):
    '''
    spaCy
    https://spacy.io/

    Minimal (buggy) implementation to show how alternate parsers can
    be added to Snorkel.
    Models for each target language needs to be downloaded using the
    following command:

    python -m spacy download en

    '''
    def __init__(self,lang='en'):
        try:
            import spacy
        except:
            raise Exception("spacy not installed. Use `pip install spacy`.")
        super(SpaCy, self).__init__(name="spaCy")
        self.model = spacy.load('en')

    def connect(self):
        return ParserConnection(self)

    def parse(self, document, text):
        '''
        Transform spaCy output to match CoreNLP's default format
        :param document:
        :param text:
        :return:
        '''
        text = text.encode('utf-8', 'error')
        text = text.decode('utf-8')

        doc = self.model(text)
        assert doc.is_parsed

        position = 0
        for sent in doc.sents:
            parts = defaultdict(list)
            dep_order, dep_par, dep_lab = [], [], []
            for token in sent:
                parts['words'].append(str(token))
                parts['lemmas'].append(token.lemma_)
                parts['pos_tags'].append(token.tag_)
                parts['ner_tags'].append(token.ent_type_)
                parts['char_offsets'].append(token.idx)

                dep_par.append(token.head)
                dep_lab.append(token.dep_)
                #dep_order.append(deps['dependent'])

            # Add null entity array (matching null for CoreNLP)
            parts['entity_cids'] = ['O' for _ in parts['words']]
            parts['entity_types'] = ['O' for _ in parts['words']]

            # Link the sentence to its parent document object
            parts['document'] = document
            parts['text'] = sent.text

            # make char_offsets relative to start of sentence
            abs_sent_offset = parts['char_offsets'][0]
            parts['char_offsets'] = [
                p - abs_sent_offset for p in parts['char_offsets']
            ]
            parts['dep_parents'] = dep_par #sort_X_on_Y(dep_par, dep_order)
            parts['dep_labels'] = dep_lab #sort_X_on_Y(dep_lab, dep_order)
            parts['position'] = position

            # Add full dependency tree parse to document meta
            # TODO

            # Assign the stable id as document's stable id plus absolute
            # character offset
            abs_sent_offset_end = (abs_sent_offset + parts['char_offsets'][-1] +
                len(parts['words'][-1]))
            parts['stable_id'] = construct_stable_id(
                document, 'sentence', abs_sent_offset, abs_sent_offset_end
            )
            position += 1
            yield parts
