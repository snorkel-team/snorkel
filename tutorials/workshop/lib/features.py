import sys
import itertools
import numpy as np
from snorkel.features import get_span_feats


def hybrid_span_mention_ftrs(candidate, stopwords=None, window=5, max_seq_ftr_len=3,
                             opts=["lemmas", "pos_tags"],
                             use_treedlib=False):
    """

    Simple sequence-based features for relation extraction. This works
    well with relation extraction tasks that frequently have contiguous arguments,
    e.g.,
     patient is experiencing [[chest]] [[pain]] since this morning.
     patient reported [[pain]] radiating from her [[chest]] to her right arm

    :param candidate:
    :param stopwords:
    :param window:
    :param opts:
    :param use_treedlib:
    :return:
    """
    args = candidate.get_contexts()
    s = candidate.get_parent()
    offsets = list(itertools.chain.from_iterable([[a.get_word_start(), a.get_word_end()] for a in args]))
    start, end = min(offsets), max(offsets)
    head = candidate[0] if candidate[0].char_start < candidate[1].char_start else candidate[1]
    tail = candidate[0] if candidate[0] != head else candidate[1]

    left = np.array(range(0, start))
    inbtwn = np.array(range(head.get_word_end() + 1, tail.get_word_start()))
    right = np.array(range(end + 1, len(s.words)))

    for attrib in opts:
        try:
            ftr_name = attrib.upper()
            if left.size != 0:
                l_ftrs = np.array(s.__dict__[attrib])[left][-max(window, 0):]
                for i, f in enumerate(l_ftrs):
                    yield (u"WIN_LEFT_{}[{}]".format(ftr_name, f), 1)
                for f in get_ngrams(l_ftrs):
                    yield (u"WIN_LEFT_SEQ_{}[{}]".format(ftr_name, f), 1)

            if inbtwn.size != 0:
                b_ftrs = np.array(s.__dict__[attrib])[inbtwn]
                for i, f in enumerate(b_ftrs):
                    yield (u"BETWEEN_{}[{}]".format(ftr_name, f), 1)
                for f in get_ngrams(b_ftrs):
                    yield (u"BETWEEN_SEQ_{}[{}]".format(ftr_name, f), 1)

            if right.size != 0:
                r_ftrs = np.array(s.__dict__[attrib])[right][:min(len(s.words), window)]
                for i, f in enumerate(r_ftrs):
                    yield (u"WIN_RIGHT_{}[{}]".format(ftr_name, f), 1)
                for f in get_ngrams(r_ftrs):
                    yield (u"WIN_RIGHT_SEQ_{}[{}]".format(ftr_name, f), 1)

            yield (u"BETWEEN_LEN[{}]".format(get_bin(len(inbtwn))), 1)

        except Exception as e:
            print>>sys.stderr, u"Featurization Error", e

    if use_treedlib:
        for f in get_span_feats(candidate):
            yield f

def get_bin(v, bins=[0, 1, 2, 4, 6, 8]):
    """

    :param v:
    :param bins:
    :return:
    """
    v = abs(v)
    for i in range(len(bins) - 1):
        if v >= bins[i] and v < bins[i + 1]:
            return "{}:{}".format(bins[i], bins[i + 1])

    return "{}:".format(bins[-1])

def get_ngrams(s, max_len=3):
    """
    Generate sliding ngram sequences

    :param s:
    :param max_len:
    :return:
    """
    terms = {}
    for i in range(len(s)):
        for j in range(i + 2, len(s) + 1):
            terms[" ".join(s[i:min(j, i + max_len)])] = 1
    return terms.keys()