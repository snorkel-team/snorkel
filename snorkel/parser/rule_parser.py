from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import re
import pkg_resources
from pathlib import Path
from collections import defaultdict
from snorkel.models import construct_stable_id
from snorkel.parser.parser import Parser, ParserConnection

try:
    import spacy
    from spacy.cli import download
    from spacy import util
except:
    raise Exception("spacy not installed. Use `pip install spacy`.")

class Tokenizer(object):
    '''
    Interface for rule-based tokenizers
    '''
    def apply(self,s):
        raise NotImplementedError()

class RegexTokenizer(Tokenizer):
    '''
    Regular expression tokenization.
    '''
    def __init__(self, rgx="\s+"):
        super(RegexTokenizer, self).__init__()
        self.rgx = re.compile(rgx)

    def apply(self,s):
        '''

        :param s:
        :return:
        '''
        tokens = []
        offset = 0
        # keep track of char offsets
        for t in self.rgx.split(s):
            while offset < len(s) and t != s[offset:offset+len(t)]:
                offset += 1
            tokens += [(t,offset)]
            offset += len(t)
        return tokens

class SpacyTokenizer(Tokenizer):
    '''
    Only use spaCy's tokenizer functionality
    '''
    def __init__(self, lang='en'):
        super(SpacyTokenizer, self).__init__()
        self.lang = lang
        self.model = SpacyTokenizer.load_lang_model(lang)

    def apply(self, s):
        doc = self.model.tokenizer(s)
        return [(t.text, t.idx) for t in doc]

    @staticmethod
    def is_package(name):
        """Check if string maps to a package installed via pip.
        name (unicode): Name of package.
        RETURNS (bool): True if installed package, False if not.

        From https://github.com/explosion/spaCy/blob/master/spacy/util.py

        """
        name = name.lower()  # compare package name against lowercase name
        packages = pkg_resources.working_set.by_key.keys()
        for package in packages:
            if package.lower().replace('-', '_') == name:
                return True
            return False

    @staticmethod
    def model_installed(name):
        '''
        Check if spaCy language model is installed

        From https://github.com/explosion/spaCy/blob/master/spacy/util.py

        :param name:
        :return:
        '''
        data_path = util.get_data_path()
        if not data_path or not data_path.exists():
            raise IOError("Can't find spaCy data path: %s" % str(data_path))
        if name in set([d.name for d in data_path.iterdir()]):
            return True
        if SpacyTokenizer.is_package(name): # installed as package
            return True
        if Path(name).exists(): # path to model data directory
            return True
        return False

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
        if SpacyTokenizer.model_installed(lang):
            model = spacy.load(lang)
        else:
            download(lang)
            model = spacy.load(lang)
        return model


class RuleBasedParser(Parser):
    '''
    Simple, rule-based parser that requires a functions for
     1) detecting sentence boundaries
     2) tokenizing
    '''
    def __init__(self, tokenizer=None, sent_boundary=None):

        super(RuleBasedParser, self).__init__(name="rules")
        self.tokenizer = tokenizer if tokenizer else SpacyTokenizer("en")
        self.sent_boundary = sent_boundary if sent_boundary else RegexTokenizer("[\n\r]+")

    def to_unicode(self, text):

        text = text.encode('utf-8', 'error')
        # Python 2
        try:
            text = text.decode('string_escape', errors='ignore')
            text = text.decode('utf-8')
        # Python 3
        except LookupError:
            text = text.decode('unicode_escape', errors='ignore')
        return text

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

        offset, position = 0, 0
        sentences = self.sent_boundary.apply(text)

        for sent,sent_offset in sentences:
            parts = defaultdict(list)
            tokens = self.tokenizer.apply(sent)
            if not tokens:
                continue

            parts['words'], parts['char_offsets'] = list(zip(*tokens))
            parts['abs_char_offsets'] = [idx + offset for idx in parts['char_offsets']]
            parts['lemmas'] = []
            parts['pos_tags'] = []
            parts['ner_tags'] = []
            parts['dep_parents'] = []
            parts['dep_labels'] = []
            parts['position'] = position

            position += 1
            offset += len(sent)

            # Link the sentence to its parent document object
            parts['document'] = document
            parts['text'] = sent

            # Add null entity array (matching null for CoreNLP)
            parts['entity_cids'] = ['O' for _ in parts['words']]
            parts['entity_types'] = ['O' for _ in parts['words']]

            # Assign the stable id as document's stable id plus absolute
            # character offset
            abs_sent_offset = parts['abs_char_offsets'][0]
            abs_sent_offset_end = abs_sent_offset + parts['char_offsets'][-1] + len(parts['words'][-1])
            if document:
                parts['stable_id'] = construct_stable_id(document, 'sentence', abs_sent_offset, abs_sent_offset_end)

            yield parts
