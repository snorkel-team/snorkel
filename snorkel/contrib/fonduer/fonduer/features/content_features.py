from builtins import str
from builtins import range
import sys
import os

from snorkel.models import TemporarySpan
from tree_structs import corenlp_to_xmltree
from snorkel.utils import get_as_dict
from ..config import settings
from ..lf_helpers import get_left_ngrams, get_right_ngrams, tokens_to_ngrams

sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator

DEF_VALUE = 1

unary_ddlib_feats = {}
unary_word_feats = {}
unary_tdl_feats = {}
binary_tdl_feats = {}


def get_content_feats(candidates):
    candidates = candidates if isinstance(candidates, list) else [candidates]
    for candidate in candidates:
        args = candidate.get_contexts()
        if not (isinstance(args[0], TemporarySpan)):
            raise ValueError("Accepts Span-type arguments, %s-type found." % type(candidate))

        # Unary candidates
        if len(args) == 1:
            span = args[0]
            if span.sentence.is_lingual():
                get_tdl_feats = compile_entity_feature_generator()
                sent = get_as_dict(span.parent)
                xmltree = corenlp_to_xmltree(sent)
                sidxs = list(range(span.get_word_start(), span.get_word_end() + 1))
                if len(sidxs) > 0:
                    # Add DDLIB entity features
                    for f in get_ddlib_feats(span, sent, sidxs):
                        yield candidate.id, 'DDL_' + f, DEF_VALUE
                    # Add TreeDLib entity features
                    if span.stable_id not in unary_tdl_feats:
                            unary_tdl_feats[span.stable_id] = set()
                            for f in get_tdl_feats(xmltree.root, sidxs):
                                unary_tdl_feats[span.stable_id].add(f)
                    for f in unary_tdl_feats[span.stable_id]:
                        yield candidate.id, 'TDL_' + f, DEF_VALUE
            else:
                for f in get_word_feats(span):
                    yield candidate.id, 'BASIC_' + f, DEF_VALUE

        # Binary candidates
        elif len(args) == 2:
            span1, span2 = args
            if span1.sentence.is_lingual() and span2.sentence.is_lingual():
                get_tdl_feats = compile_relation_feature_generator()
                sent1 = get_as_dict(span1.sentence)
                sent2 = get_as_dict(span2.sentence)
                xmltree = corenlp_to_xmltree(get_as_dict(span1.sentence))
                s1_idxs = list(range(span1.get_word_start(), span1.get_word_end() + 1))
                s2_idxs = list(range(span2.get_word_start(), span2.get_word_end() + 1))
                if len(s1_idxs) > 0 and len(s2_idxs) > 0:

                    # Add DDLIB entity features for relation
                    for f in get_ddlib_feats(span1, sent1, s1_idxs):
                        yield candidate.id, 'DDL_e1_' + f, DEF_VALUE

                    for f in get_ddlib_feats(span2, sent2, s2_idxs):
                        yield candidate.id, 'DDL_e2_' + f, DEF_VALUE

                    # Add TreeDLib relation features
                    if candidate.id not in binary_tdl_feats:
                            binary_tdl_feats[candidate.id] = set()
                            for f in get_tdl_feats(xmltree.root, s1_idxs, s2_idxs):
                                binary_tdl_feats[candidate.id].add(f)
                    for f in binary_tdl_feats[candidate.id]:
                        yield candidate.id, 'TDL_' + f, DEF_VALUE
            else:
                for f in get_word_feats(span1):
                    yield candidate.id, 'BASIC_e1_' + f, DEF_VALUE

                for f in get_word_feats(span2):
                    yield candidate.id, 'BASIC_e2_' + f, DEF_VALUE

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


def get_ddlib_feats(span, context, idxs):
    """
    Minimalist port of generic mention features from ddlib
    """

    if span.stable_id not in unary_ddlib_feats:
        unary_ddlib_feats[span.stable_id] = set()

        for seq_feat in _get_seq_features(context, idxs):
            unary_ddlib_feats[span.stable_id].add(seq_feat)

        for window_feat in _get_window_features(context, idxs):
            unary_ddlib_feats[span.stable_id].add(window_feat)

    for f in unary_ddlib_feats[span.stable_id]:
        yield f

def _get_seq_features(context, idxs):
    yield "WORD_SEQ_[" + " ".join(context['words'][i] for i in idxs) + "]"
    yield "LEMMA_SEQ_[" + " ".join(context['lemmas'][i] for i in idxs) + "]"
    yield "POS_SEQ_[" + " ".join(context['pos_tags'][i] for i in idxs) + "]"
    yield "DEP_SEQ_[" + " ".join(context['dep_labels'][i] for i in idxs) + "]"


def _get_window_features(context, idxs, window=settings.featurization.content.window_feature.size,
                         combinations=settings.featurization.content.window_feature.combinations,
                         isolated=settings.featurization.content.window_feature.isolated):
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
                yield "W_LEMMA_L_" + str(i + 1) + "_R_" + str(
                    j + 1) + "_[" + curr_left_lemmas + "]_[" + curr_right_lemmas + "]"
                yield "W_POS_L_" + str(i + 1) + "_R_" + str(
                    j + 1) + "_[" + curr_left_pos_tags + "]_[" + curr_right_pos_tags + "]"


def get_word_feats(span):
    attrib = 'words'

    if span.stable_id not in unary_word_feats:
        unary_word_feats[span.stable_id] = set()

        for ngram in tokens_to_ngrams(span.get_attrib_tokens(attrib), n_min=1, n_max=2):
            feature = "CONTAINS_%s_[%s]" % (attrib.upper(), ngram)
            unary_word_feats.add(feature)

        for ngram in get_left_ngrams(span,
                                     window=settings.featurization.content.word_feature.window,
                                     n_max=2, attrib=attrib):
            feature = "LEFT_%s_[%s]" % (attrib.upper(), ngram)
            unary_word_feats.add(feature)

        for ngram in get_right_ngrams(span,
                                      window=settings.featurization.content.word_feature.window,
                                      n_max=2, attrib=attrib):
            feature = "RIGHT_%s_[%s]" % (attrib.upper(), ngram)
            unary_word_feats.add(feature)

    for f in unary_word_feats[span.stable_id]:
        yield f
