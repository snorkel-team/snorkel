from ..lf_helpers import *

FEAT_PRE = 'VIZ_'
DEF_VALUE = 1


def get_visual_feats(candidate):
    args = candidate.get_contexts()
    if not (isinstance(args[0], TemporarySpan)):
        raise ValueError("Accepts Span-type arguments, %s-type found." % type(candidate))

    # Unary candidates
    if len(args) == 1:
        span = args[0]
        # Add VisualLib entity features (if applicable)
        if span.is_visual():
            for f, v in vizlib_unary_features(span):
                yield FEAT_PRE + f, v

    # Binary candidates
    elif len(args) == 2:
        span1, span2 = args
        # Add VisualLib entity features (if applicable)
        if span1.is_visual() or span2.is_visual():
            for f, v in vizlib_binary_features(span1, span2):
                yield FEAT_PRE + f, v
    else:
        raise NotImplementedError("Only handles unary and binary candidates currently")


def vizlib_unary_features(span):
    """
    Visual-related features for a single span
    """
    if not span.is_visual(): return

    for f in get_visual_aligned_lemmas(span):
        yield 'ALIGNED_' + f, DEF_VALUE

    for page in set(span.get_attrib_tokens('page')):
        yield "PAGE_[%d]" % page, DEF_VALUE


def vizlib_binary_features(span1, span2):
    """
    Visual-related features for a pair of spans
    """
    for feat, v in vizlib_unary_features(span1):
        yield "e1_" + feat, v
    for feat, v in vizlib_unary_features(span2):
        yield "e2_" + feat, v

    if same_page((span1, span2)):
        yield "SAME_PAGE", DEF_VALUE

        if is_horz_aligned((span1, span2)):
            yield "HORZ_ALIGNED", DEF_VALUE

        if is_vert_aligned((span1, span2)):
            yield "VERT_ALIGNED", DEF_VALUE

        if is_vert_aligned_left((span1, span2)):
            yield "VERT_ALIGNED_LEFT", DEF_VALUE

        if is_vert_aligned_right((span1, span2)):
            yield "VERT_ALIGNED_RIGHT", DEF_VALUE

        if is_vert_aligned_center((span1, span2)):
            yield "VERT_ALIGNED_CENTER", DEF_VALUE
