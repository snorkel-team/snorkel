from collections import namedtuple
import re

# fields = ['words', 'char_offsets', 'word_offsets', 'pos_tags', 'ner_tags', 'entity_types', 'text']
# Phrase = namedtuple('Phrase', fields)

inequalities = {
    '.lt': lambda x, y: x < y,
    '.leq': lambda x, y: x <= y,
    '.eq': lambda x, y: x == y, # if the index is not exact, continue
    '.geq': lambda x, y: x >= y,
    '.gt': lambda x, y: x > y,
}


class Phrase(object):
    fields = ['text', 'words', 'char_offsets', 'pos_tags', 'ner_tags', 'entity_types']

    def __init__(self, sentence=None):
        for field in self.fields:
            setattr(self, field, getattr(sentence, field) if sentence else None)
            # NOTE: I believe all candidate-related fields are already unicode.
            # value = getattr(sentence, field) if sentence else None
            # # Convert all str inputs into unicode
            # if isinstance(value, str):
            #     value = value.decode('utf-8')
            # elif isinstance(value, list) and len(value) and isinstance(value[0], str):
            #     value = [w.decode('utf-8') for w in value]
            # setattr(self, field, value)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
        else:
            assert isinstance(key, int)
            start = key
            stop = key + 1
        p = Phrase()
        text_start = self.char_offsets[start]
        text_stop = self.char_offsets[stop] if stop < len(self.char_offsets) else None
        p.text = self.text[text_start:text_stop]
        p.words = self.words[start:stop]
        p.char_offsets = self.char_offsets[start:stop]
        p.pos_tags = self.pos_tags[start:stop]
        p.ner_tags = self.ner_tags[start:stop]
        p.entity_types = self.entity_types[start:stop]
        return p

    def __len__(self):
        return len(self.words)

    def __repr__(self):
        return 'Phrase("{}" : {} tokens)'.format(self.text.strip(), len(self.words))


def index_word(string, index):
    words = string.split()
    return _index_wordlist(words, index)


def index_phrase(phrase, index):
    words = phrase.words
    return _index_wordlist(words, index)


def _index_wordlist(wordlist, index):
    if len(wordlist) == 0:
        return ''
    if index > 0:
        index = index - 1
    elif index < 0:
        index = len(wordlist) + index
    return wordlist[max(0, min(index, len(wordlist) - 1))]


def phrase_filter(phr, field, val):
    if field == 'words':
        return [key for key in getattr(phr, field) if re.match(val, key)]
    elif field == 'chars':
        return [c for c in phr.text.strip()]
    else: # NER or POS
        # Don't count a two-token person (John Smith) as two people
        results = []
        on = False
        for i, key in enumerate(getattr(phr, field)):
            if re.match(val, key):
                if not on:
                    text_start = phr.char_offsets[i]
                    results.append(phr.words[i])
                    on = True
                else:
                    text_stop = phr.char_offsets[i + 1] if i + 1 < len(phr.char_offsets) else None
                    results[-1] = phr.text[text_start:text_stop]
            else:
                on = False
        return results


def get_left_phrase(span, cmp='.gt', num=0, unit='words'):
    phrase = Phrase(span.get_parent())
    k = span.get_word_start()
    indices = []
    for i in xrange(k):
        if unit == 'words':
            if inequalities[cmp](-i, -k + num):
                indices.append(i)
        elif unit == 'chars':
            I = span.word_to_char_index(i)
            K = span.word_to_char_index(k)
            if inequalities[cmp](-I, -K + num):
                indices.append(i)
        else:
            raise Exception("Expected unit in ('words', 'chars'), got '{}'".format(unit))
    if indices:
        return phrase[min(indices):max(indices) + 1]
    else:
        return phrase[0:0] # empty phrase


def get_right_phrase(span, cmp='.gt', num=0, unit='words'):
    phrase = Phrase(span.get_parent())
    k = span.get_word_end()
    indices = []
    for i in xrange(k + 1, len(phrase)):
        if unit == 'words':
            if inequalities[cmp](i, k + num):
                indices.append(i)
        elif unit == 'chars':
            I = span.word_to_char_index(i)
            K = span.word_to_char_index(k)
            if inequalities[cmp](I, K + num):
                indices.append(i)            
        else:        
            raise Exception("Expected unit in ('words', 'chars'), got '{}'".format(unit))
    if indices:
        return phrase[min(indices):max(indices) + 1]
    else:
        return phrase[0:0]

    
def get_within_phrase(span, num=0, unit='words'):
    phrase = Phrase(span.get_parent())
    if unit == 'words':
        j = span.get_word_start()
        k = span.get_word_end()
        return phrase[max(0, j - num):min(k + num + 1, len(phrase))]
    elif unit == 'chars':
        raise NotImplementedError
    else:        
        raise Exception("Expected unit in ('words', 'chars'), got '{}'".format(unit))


def get_between_phrase(span1, span2):
    phrase = Phrase(span1.get_parent())
    if span1.char_start > span2.char_start:
        span1, span2 = span2, span1
    i = span1.get_word_end()
    j = span2.get_word_start()
    return phrase[i + 1:j]


def get_sentence_phrase(span):
    return Phrase(span.get_parent())


helpers = {
    'index_word': index_word,
    'index_phrase': index_phrase,
    'phrase_filter': phrase_filter,
    'get_left_phrase': get_left_phrase,
    'get_right_phrase': get_right_phrase,
    'get_within_phrase': get_within_phrase,
    'get_between_phrase': get_between_phrase,
    'get_sentence_phrase': get_sentence_phrase,
}