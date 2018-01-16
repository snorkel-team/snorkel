from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from collections import defaultdict
from snorkel.models import construct_stable_id
from snorkel.parser import Parser, ParserConnection

try:
    import spacy
    from spacy.cli import download
    from spacy import util
    from spacy.deprecated import resolve_model_name
except:
    raise Exception("spaCy not installed. Use `pip install spacy`.")


class Spacy(Parser):
    '''
    spaCy
    https://spacy.io/

    Models for each target language needs to be downloaded using the
    following command:

    python -m spacy download en

    Default named entity types

    PERSON	    People, including fictional.
    NORP	    Nationalities or religious or political groups.
    FACILITY	Buildings, airports, highways, bridges, etc.
    ORG	        Companies, agencies, institutions, etc.
    GPE	        Countries, cities, states.
    LOC	        Non-GPE locations, mountain ranges, bodies of water.
    PRODUCT	    Objects, vehicles, foods, etc. (Not services.)
    EVENT	    Named hurricanes, battles, wars, sports events, etc.
    WORK_OF_ART	Titles of books, songs, etc.
    LANGUAGE	Any named language.

    DATE	    Absolute or relative dates or periods.
    TIME	    Times smaller than a day.
    PERCENT	    Percentage, including "%".
    MONEY	    Monetary values, including unit.
    QUANTITY	Measurements, as of weight or distance.
    ORDINAL	    "first", "second", etc.
    CARDINAL	Numerals that do not fall under another type.

    '''
    def __init__(self, annotators=['tagger', 'parser', 'entity'],
                 lang='en', num_threads=1, verbose=False):

        super(Spacy, self).__init__(name="spacy")
        self.model = Spacy.load_lang_model(lang)
        self.num_threads = num_threads

        self.pipeline = []
        for proc in annotators:
            self.pipeline += [self.model.__dict__[proc]]

    @staticmethod
    def model_installed(name):
        '''
        Check if spaCy language model is installed
        :param name:
        :return:
        '''
        data_path = util.get_data_path()
        model_name = resolve_model_name(name)
        model_path = data_path / model_name
        return model_path.exists()

    @staticmethod
    def load_lang_model(lang):
        '''
        Load spaCy language model or download if
        model is available and not installed

        Currenty supported spaCy languages

        en English (50MB)
        de German (645MB)
        fr French (1.33GB)
        es Spanish (377MB)

        :param lang:
        :return:
        '''
        if not Spacy.model_installed(lang):
            download(lang)
        return spacy.load(lang)

    def connect(self):
        return ParserConnection(self)

    def parse(self, document, text):
        '''
        Transform spaCy output to match CoreNLP's default format
        :param document:
        :param text:
        :return:
        '''
        text = self.to_unicode(text)

        doc = self.model.tokenizer(text)
        for proc in self.pipeline:
            proc(doc)
        assert doc.is_parsed

        position = 0
        for sent in doc.sents:
            parts = defaultdict(list)
            text = sent.text

            for i,token in enumerate(sent):
                parts['words'].append(str(token))
                parts['lemmas'].append(token.lemma_)
                parts['pos_tags'].append(token.tag_)
                parts['ner_tags'].append(token.ent_type_ if token.ent_type_ else 'O')
                parts['char_offsets'].append(token.idx)
                parts['abs_char_offsets'].append(token.idx)
                head_idx = 0 if token.head is token else token.head.i - sent[0].i + 1
                parts['dep_parents'].append(head_idx)
                parts['dep_labels'].append(token.dep_)

            # Add null entity array (matching null for CoreNLP)
            parts['entity_cids'] = ['O' for _ in parts['words']]
            parts['entity_types'] = ['O' for _ in parts['words']]

            # make char_offsets relative to start of sentence
            parts['char_offsets'] = [
                p - parts['char_offsets'][0] for p in parts['char_offsets']
            ]
            parts['position'] = position

            # Link the sentence to its parent document object
            parts['document'] = document
            parts['text'] = text

            # Add null entity array (matching null for CoreNLP)
            parts['entity_cids'] = ['O' for _ in parts['words']]
            parts['entity_types'] = ['O' for _ in parts['words']]

            # Assign the stable id as document's stable id plus absolute
            # character offset
            abs_sent_offset = parts['abs_char_offsets'][0]
            abs_sent_offset_end = abs_sent_offset + parts['char_offsets'][-1] + len(parts['words'][-1])
            if document:
                parts['stable_id'] = construct_stable_id(document, 'sentence', abs_sent_offset, abs_sent_offset_end)

            position += 1

            yield parts
