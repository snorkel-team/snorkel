from collections import namedtuple

# VERSION 1

# def get_left_tokens(span, attrib='words'):
#     """
#     Returns the tokens between span0 and span1
#     """
#     i = span.get_word_start()
#     return span.get_parent()._asdict()[attrib][:i][::-1]

# def get_right_tokens(span, attrib='words'):
#     """
#     Returns the tokens between span0 and span1
#     """
#     i = span.get_word_end()
#     return span.get_parent()._asdict()[attrib][i+1:]

# def get_between_tokens(span0, span1, attrib='words'):
#     """
#     Returns the tokens between span0 and span1
#     """
#     if span0.get_word_start() < span1.get_word_start():
#         left_span = span0
#         dist_btwn = span1.get_word_start() - span0.get_word_end() - 1
#     else:
#         left_span = span1
#         dist_btwn = span0.get_word_start() - span1.get_word_end() - 1
#     i = left_span.get_word_end()
#     return left_span.get_parent()._asdict()[attrib][i+1:i+1+dist_btwn]

# def get_sentence_tokens(span, attrib='words'):
#     """
#     Returns the tokens in the sentence of the span
#     """
#     return span.get_parent()._asdict()[attrib]

# VERSION 2

# def get_left_tokens(span):
    # i = span.get_word_start()
    # sent = span.get_parent()
    # partial = {}
    # for key, values in sent._asdict().iteritems():
    #     if key in relevant:
    #         partial[key] = values[:i]
    # return partial

# def get_right_tokens(span):
#     i = span.get_word_start()
#     sent = span.get_parent()
#     partial = {}
#     for key, values in sent._asdict().iteritems():
#         if key in relevant:
#             partial[key] = values[i+1:]
#     return partial

# def get_between_tokens(span0, span1):
#     if span0.get_word_start() < span1.get_word_start():
#         left_span = span0
#         dist_btwn = span1.get_word_start() - span0.get_word_end() - 1
#     else:
#         left_span = span1
#         dist_btwn = span0.get_word_start() - span1.get_word_end() - 1
#     i = left_span.get_word_end()
#     sent = span0.get_parent()
#     partial = {}
#     for key, values in sent._asdict().iteritems():
#         if key in relevant:
#             partial[key] = values[i+1:i+1+dist_btwn]
#     return partial

# def get_sentence_tokens(span):
#     sent = span.get_parent()
#     partial = {}
#     for key, values in sent._asdict().iteritems():
#         if key in relevant:
#             partial[key] = values
#     return partial

# VERSION 3
# fields = ['words', 'char_offsets', 'pos_tags', 'ner_tags', 'entity_types']
# Token = namedtuple('Token', fields)
# fields = ['words', 'word_offsets', 'char_offsets', 'pos_tags', 'ner_tags', 'entity_types', 'text']
# Phrase = namedtuple('Phrase', fields)

# def get_phrase_from_span(span):
#     contents = []
#     for f in fields:
#         if f=='word_offsets':
#             word_indices = range(span.get_word_start(), span.get_word_end() + 1)
#             contents.append(word_indices)
#         elif f=='char_offsets':
#             contents.append([span.word_to_char_index(wi) for wi in word_indices])
#         else:
#             contents.append(span.get_attrib_tokens(a=f))
#     return Phrase(*contents)

# def get_phrase_from_text(sentence, text):
#     # TODO: precompute these and hash them with each sentence for speed?
#     import pdb; pdb.set_trace()
#     sent_dict = sentence._asdict()
#     sent_text = sent_dict['text']
#     sent_tokens = sent_dict['words']
#     num_sent_tokens = len(sent_tokens)
#     num_text_tokens = len(text.split())
#     char_starts = []
#     print sent_text
#     print text
#     for L in range(num_text_tokens, num_text_tokens + 2):
#         for i in range(0, len_sent_tokens - L + 1):
#             char_start = sent_dict['char_offsets'][i]
#             char_end = char_start + len(sent_dict['words'][i+L])
#             print sent_text[char_start:char_end]
#             # if sent_text[char_start:char_end]
    
#     # TODO: get char_starts of tokens/phrases only (don't just search in text)
#     # ci = -1
#     # while ci < sent_len:
#     #     ci = sent_text[ci+1:]
#     #     if ci == -1:
#     #         ci = sent_len
#     #     else:
#     #         char_starts.append(ci)
#     return [] # return a list

# def get_right_phrases(span, n_min=1, n_max=3):
#     phrases = []
#     k = span.get_word_start()
#     sent = span.get_parent()._asdict()
#     for i in range(k+1, len(sent['words'])):
#         phrases.append(Phrase(*[[sent[field][i]] for field in fields]))
#     return phrases

fields = ['words', 'char_offsets', 'word_offsets', 'pos_tags', 'ner_tags', 'entity_types', 'text']
Phrase = namedtuple('Phrase', fields)

inequalities = {
    '.lt': lambda x, y: x < y,
    '.leq': lambda x, y: x <= y,
    '.eq': lambda x, y: x == y, # if the index is not exact, continue
    '.geq': lambda x, y: x >= y,
    '.gt': lambda x, y: x > y,
}

def build_phrase(sent, i, L):
    contents = []
    for f in fields:
        if f == 'word_offsets':
            contents.append(range(i, i+L))
        elif f == 'text':
            contents.append(sent['text'][sent['char_offsets'][i]:sent['char_offsets'][i+L]].strip())
        else:
            contents.append(sent[f][i:i+L])
    return Phrase(*contents)

def get_left_phrases(span, cmp='.gt', num=0, unit='words', n_min=1, n_max=4):
    """
    "at least 40 characters to the left of X" => (X, .geq, 40, chars)
    Note: Bases distances on the starts of words/chars
    """
    phrases = []
    k = span.get_word_start()
    sent = span.get_parent()._asdict()
    for L in range(n_min, n_max+1): # how long is the n-gram
        for i in range(0, k-L+1): # where does it start
            if unit=='words':
                if not inequalities[cmp](-i, -k + num):
                    continue
            else:
                I = span.word_to_char_index(i)
                K = span.word_to_char_index(k)
                if not inequalities[cmp](-I, -K + num):
                    continue
            phrases.append(build_phrase(sent, i, L))
    return phrases

def get_right_phrases(span, cmp='.gt', num=0, unit='words', n_min=1, n_max=4):
    phrases = []
    k = span.get_word_end()
    sent = span.get_parent()._asdict()
    for L in range(n_min, n_max+1):
        for i in range(k+1, len(sent['words'])-L):
            if unit=='words':
                if not inequalities[cmp](i, k + num):
                    continue
            else:
                I = span.word_to_char_index(i)
                K = span.word_to_char_index(k) + len(sent['words'][k])
                if not inequalities[cmp](I, K + num):
                    continue
            phrases.append(build_phrase(sent, i, L))
    return phrases

def get_within_phrases(span):
    pass
#     phrases = []
#     ks = span.get_word_start()
#     ke = span.get_word_end()
#     sent = span.get_parent()._asdict()
#     for L in range(n_min, n_max+1):
#         for i in range(k+1, len(sent['words'])-L):
#             if unit=='words':
#                 if not inequalities[cmp](i, k + num):
#                     continue
#             else:
#                 I = span.word_to_char_index(i)
#                 K = span.word_to_char_index(k) + len(sent['words'][k])
#                 if not inequalities[cmp](I, K + num):
#                     continue
#             phrases.append(build_phrase(sent, i, L))
#     return phrases    

def get_between_phrases(span0, span1, n_min=1, n_max=4):
    phrases = []
    if span0.get_word_start() < span1.get_word_start():
        left_span = span0
        dist_btwn = span1.get_word_start() - span0.get_word_end() - 1
    else:
        left_span = span1
        dist_btwn = span0.get_word_start() - span1.get_word_end() - 1
    k = left_span.get_word_end()
    sent = span0.get_parent()._asdict()
    for L in range(n_min, n_max+1):
        for i in range(k+1, k+dist_btwn-L+2):
            phrases.append(build_phrase(sent, i, L))
    return phrases

def get_sentence_phrases(span, n_min=1, n_max=4):
    phrases = []
    k = span.get_word_start()
    sent = span.get_parent()._asdict()
    for L in range(n_min, n_max+1):
        for i in range(0, len(sent['words'])-L):
            phrases.append(build_phrase(sent, i, L))
    # import pdb; pdb.set_trace()
    return phrases

def lf_helpers():
    return {
            # 'get_phrase_from_text': get_phrase_from_text,
            # 'get_phrase_from_span': get_phrase_from_span,
            'get_left_phrases': get_left_phrases,
            'get_right_phrases': get_right_phrases,
            'get_within_phrases': get_within_phrases,
            'get_between_phrases': get_between_phrases,
            'get_sentence_phrases': get_sentence_phrases,
            }