import sys
import os

from snorkel.models import TemporarySpan
from tree_structs import corenlp_to_xmltree
from snorkel.utils import get_as_dict

sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator

DEF_VALUE = 1


def get_content_feats(candidate):
    args = candidate.get_arguments()
    if not (isinstance(args[0], TemporarySpan)):
        raise ValueError("Accepts Span-type arguments, %s-type found." % type(candidate))

    # Unary candidates
    if len(args) == 1:
        get_tdl_feats = compile_entity_feature_generator()
        span = args[0]
        if not span.is_lingual(): return
        sent = get_as_dict(span.parent)
        xmltree = corenlp_to_xmltree(sent)
        sidxs = range(span.get_word_start(), span.get_word_end() + 1)
        if len(sidxs) > 0:
            # Add DDLIB entity features
            for f in get_ddlib_feats(sent, sidxs):
                yield 'DDL_' + f, DEF_VALUE
            # Add TreeDLib entity features
            for f in get_tdl_feats(xmltree.root, sidxs):
                yield 'TDL_' + f, DEF_VALUE
    # Binary candidates
    elif len(args) == 2:
        get_tdl_feats = compile_relation_feature_generator()
        span1, span2 = args
        # TODO: check if span.is_lingual() is True for each span
        xmltree = corenlp_to_xmltree(get_as_dict(span1.parent))
        s1_idxs = range(span1.get_word_start(), span1.get_word_end() + 1)
        s2_idxs = range(span2.get_word_start(), span2.get_word_end() + 1)
        if len(s1_idxs) > 0 and len(s2_idxs) > 0:
            # Apply TreeDLib relation features
            for f in get_tdl_feats(xmltree.root, s1_idxs, s2_idxs):
                yield 'TDL_' + f, DEF_VALUE

            # TODO: add DDLib features for binary relations

    else:
        raise NotImplementedError("Only handles unary and binary candidates currently")


def compile_entity_feature_generator():
    """
    Given optional arguments, returns a generator function which accepts an xml root
    and a list of indexes for a mention, and will generate relation features for this entity
    """

    BASIC_ATTRIBS_REL = ['lemma', 'dep_label']

    m = Mention(0)

    # Basic relation feature templates
    temps = [
        [Indicator(m, a) for a in BASIC_ATTRIBS_REL],
        Indicator(m, 'dep_label,lemma'),
        # The *first element on the* path to the root: ngram lemmas along it
        Ngrams(Parents(m, 3), 'lemma', (1, 3)),
        Ngrams(Children(m), 'lemma', (1, 3)),
        # The siblings of the mention
        [LeftNgrams(LeftSiblings(m), a) for a in BASIC_ATTRIBS_REL],
        [RightNgrams(RightSiblings(m), a) for a in BASIC_ATTRIBS_REL]
    ]

    # return generator function
    return Compile(temps).apply_mention


def get_ddlib_feats(context, idxs):
    """
    Minimalist port of generic mention features from ddlib
    """

    for seq_feat in _get_seq_features(context, idxs):
        yield seq_feat

    for window_feat in _get_window_features(context, idxs):
        yield window_feat

    if context['words'][idxs[0]][0].isupper():
        yield "STARTS_WITH_CAPITAL"

    yield "LENGTH_{}".format(len(idxs))


def _get_seq_features(context, idxs):
    yield "WORD_SEQ_[" + " ".join(context['words'][i] for i in idxs) + "]"
    yield "LEMMA_SEQ_[" + " ".join(context['lemmas'][i] for i in idxs) + "]"
    yield "POS_SEQ_[" + " ".join(context['pos_tags'][i] for i in idxs) + "]"
    yield "DEP_SEQ_[" + " ".join(context['dep_labels'][i] for i in idxs) + "]"


def _get_window_features(context, idxs, window=3, combinations=True, isolated=True):
    left_lemmas = []
    left_pos_tags = []
    right_lemmas = []
    right_pos_tags = []
    try:
        for i in range(1, window + 1):
            lemma = context['lemmas'][idxs[0] - i]
            try:
                float(lemma)
                lemma = "_NUMBER"
            except ValueError:
                pass
            left_lemmas.append(lemma)
            left_pos_tags.append(context['pos_tags'][idxs[0] - i])
    except IndexError:
        pass
    left_lemmas.reverse()
    left_pos_tags.reverse()
    try:
        for i in range(1, window + 1):
            lemma = context['lemmas'][idxs[-1] + i]
            try:
                float(lemma)
                lemma = "_NUMBER"
            except ValueError:
                pass
            right_lemmas.append(lemma)
            right_pos_tags.append(context['pos_tags'][idxs[-1] + i])
    except IndexError:
        pass
    if isolated:
        for i in range(len(left_lemmas)):
            yield "W_LEFT_" + str(i + 1) + "_[" + " ".join(left_lemmas[-i - 1:]) + "]"
            yield "W_LEFT_POS_" + str(i + 1) + "_[" + " ".join(left_pos_tags[-i - 1:]) + "]"
        for i in range(len(right_lemmas)):
            yield "W_RIGHT_" + str(i + 1) + "_[" + " ".join(right_lemmas[:i + 1]) + "]"
            yield "W_RIGHT_POS_" + str(i + 1) + "_[" + " ".join(right_pos_tags[:i + 1]) + "]"
    if combinations:
        for i in range(len(left_lemmas)):
            curr_left_lemmas = " ".join(left_lemmas[-i - 1:])
            try:
                curr_left_pos_tags = " ".join(left_pos_tags[-i - 1:])
            except TypeError:
                new_pos_tags = []
                for pos in left_pos_tags[-i - 1:]:
                    to_add = pos
                    if not to_add:
                        to_add = "None"
                    new_pos_tags.append(to_add)
                curr_left_pos_tags = " ".join(new_pos_tags)
            for j in range(len(right_lemmas)):
                curr_right_lemmas = " ".join(right_lemmas[:j + 1])
                try:
                    curr_right_pos_tags = " ".join(right_pos_tags[:j + 1])
                except TypeError:
                    new_pos_tags = []
                    for pos in right_pos_tags[:j + 1]:
                        to_add = pos
                        if not to_add:
                            to_add = "None"
                        new_pos_tags.append(to_add)
                    curr_right_pos_tags = " ".join(new_pos_tags)
                yield "W_LEMMA_L_" + str(i + 1) + "_R_" + str(j + 1) + "_[" + curr_left_lemmas + "]_[" + curr_right_lemmas + "]"
                yield "W_POS_L_" + str(i + 1) + "_R_" + str(j + 1) + "_[" + curr_left_pos_tags + "]_[" + curr_right_pos_tags + "]"

# TODO:
# yield "SPAN_TYPE_[%s]" % ('IMPLICIT' if isinstance(span, ImplicitSpan) else 'EXPLICIT'), 1
#    if phrase.html_tag:
#         yield u"HTML_TAG_" + phrase.html_tag, DEF_VALUE
#     # Comment out for now, we could calc it later.
#     # for attr in phrase.html_attrs:
#     #     yield u"HTML_ATTR_[" + attr + "]", DEF_V
#     # if phrase.html_anc_tags:
#     #     for tag in phrase.html_anc_tags:
#     #         yield u"HTML_ANC_TAG_[" + tag + "]", DEF_VALUE
#             # for attr in phrase.html_anc_attrs:
#             # yield u"HTML_ANC_ATTR_[" + attr + "]"
#     for attrib in ['words']:  # ,'lemmas', 'pos_tags', 'ner_tags']:
#         for ngram in span.get_attrib_tokens(attrib):
#             yield "CONTAINS_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
#         for ngram in get_left_ngrams(span, window=7, n_max=2, attrib=attrib):
#             yield "LEFT_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
#         for ngram in get_right_ngrams(span, window=7, n_max=2, attrib=attrib):
#             yield "RIGHT_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
#         if phrase.row_start is None or phrase.col_start is None:
#             for ngram in get_neighbor_phrase_ngrams(span, d=1, n_max=2, attrib=attrib):
#                 yield "NEIGHBOR_PHRASE_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE