import re
from collections import defaultdict
from snorkel.models import construct_stable_id
from snorkel.parser import Parser, ParserConnection

try:
    import spacy
    from spacy.cli import download
    from spacy import util
    from spacy.deprecated import resolve_model_name
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
            while t < len(s) and t != s[offset:len(t)]:
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
    def model_installed(name):
        '''
        Check if spaCy language model is installed
        :param name:
        :return:
        '''
        data_path = util.get_data_path()
        model_name = resolve_model_name(name)
        model_path = data_path / model_name
        if not model_path.exists():
            lang_name = util.get_lang_class(name).lang
            return False
        return True

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
        if SpacyTokenizer.model_installed:
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
        text = text.decode('string_escape', errors='ignore')
        text = text.decode('utf-8')
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

            parts['words'], parts['char_offsets'] = zip(*tokens)
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