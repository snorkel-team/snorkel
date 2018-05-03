from builtins import str
from ..lf_helpers import *

FEAT_PRE = 'STR_'
DEF_VALUE = 1

unary_strlib_feats = {}
binary_strlib_feats = {}


def get_structural_feats(candidates):
    candidates = candidates if isinstance(candidates, list) else [candidates]
    for candidate in candidates:
        args = candidate.get_contexts()
        if not (isinstance(args[0], TemporarySpan)):
            raise ValueError("Accepts Span-type arguments, %s-type found." % type(candidate))

        # Unary candidates
        if len(args) == 1:
            span = args[0]
            if span.sentence.is_structural():
                if span.stable_id not in unary_strlib_feats:
                    unary_strlib_feats[span.stable_id] = set()
                    for f, v in strlib_unary_features(span):
                        unary_strlib_feats[span.stable_id].add((f, v))

                for f, v in unary_strlib_feats[span.stable_id]:
                    yield candidate.id, FEAT_PRE + f, v


        # Binary candidates
        elif len(args) == 2:
            span1, span2 = args
            if span1.sentence.is_structural() or span2.sentence.is_structural():
                for span, pre in [(span1, "e1_"), (span2, "e2_")]:
                    if span.stable_id not in unary_strlib_feats:
                        unary_strlib_feats[span.stable_id] = set()
                        for f, v in strlib_unary_features(span):
                            unary_strlib_feats[span.stable_id].add((f, v))

                    for f, v in unary_strlib_feats[span.stable_id]:
                        yield candidate.id, FEAT_PRE + pre + f, v

                if candidate.id not in binary_strlib_feats:
                    binary_strlib_feats[candidate.id] = set()
                    for f, v in strlib_binary_features(span1, span2):
                        binary_strlib_feats[candidate.id].add((f, v))

                for f, v in binary_strlib_feats[candidate.id]:
                    yield candidate.id, FEAT_PRE + f, v
        else:
            raise NotImplementedError("Only handles unary and binary candidates currently")


def strlib_unary_features(span):
    """
    Structural-related features for a single span
    """
    if not span.sentence.is_structural(): return

    yield "TAG_" + get_tag(span), DEF_VALUE

    for attr in get_attributes(span):
        yield "HTML_ATTR_" + attr, DEF_VALUE

    yield "PARENT_TAG_" + get_parent_tag(span), DEF_VALUE

    prev_tags = get_prev_sibling_tags(span)
    if len(prev_tags):
        yield "PREV_SIB_TAG_" + prev_tags[-1], DEF_VALUE
        yield "NODE_POS_" + str(len(prev_tags) + 1), DEF_VALUE
    else:
        yield "FIRST_NODE", DEF_VALUE

    next_tags = get_next_sibling_tags(span)
    if len(next_tags):
        yield "NEXT_SIB_TAG_" + next_tags[0], DEF_VALUE
    else:
        yield "LAST_NODE", DEF_VALUE

    yield "ANCESTOR_CLASS_[%s]" % " ".join(get_ancestor_class_names(span)), DEF_VALUE

    yield "ANCESTOR_TAG_[%s]" % " ".join(get_ancestor_tag_names(span)), DEF_VALUE

    yield "ANCESTOR_ID_[%s]" % " ".join(get_ancestor_id_names(span)), DEF_VALUE


def strlib_binary_features(span1, span2):
    """
    Structural-related features for a pair of spans
    """
    yield "COMMON_ANCESTOR_[%s]" % " ".join(common_ancestor((span1, span2))), DEF_VALUE

    yield "LOWEST_ANCESTOR_DEPTH_[%d]" % lowest_common_ancestor_depth((span1, span2)), DEF_VALUE
