from ..lf_helpers import *

FEAT_PRE = 'VIZ_'
DEF_VALUE = 1

unary_vizlib_feats = {}
binary_vizlib_feats = {}


def get_visual_feats(candidates):
    candidates = candidates if isinstance(candidates, list) else [candidates]
    for candidate in candidates:
        args = candidate.get_contexts()
        if not (isinstance(args[0], TemporarySpan)):
            raise ValueError("Accepts Span-type arguments, %s-type found." % type(candidate))

        # Unary candidates
        if len(args) == 1:
            span = args[0]
            # Add VisualLib entity features (if applicable)
            if span.sentence.is_visual():
                if span.stable_id not in unary_vizlib_feats:
                    unary_vizlib_feats[span.stable_id] = set()
                    for f, v in vizlib_unary_features(span):
                        unary_vizlib_feats[span.stable_id].add((f, v))

                for f, v in unary_vizlib_feats[span.stable_id]:
                    yield candidate.id, FEAT_PRE + f, v

        # Binary candidates
        elif len(args) == 2:
            span1, span2 = args
            # Add VisualLib entity features (if applicable)
            if span1.sentence.is_visual() or span2.sentence.is_visual():
                for span, pre in [(span1, "e1_"), (span2, "e2_")]:
                    if span.stable_id not in unary_vizlib_feats:
                        unary_vizlib_feats[span.stable_id] = set()
                        for f, v in vizlib_unary_features(span):
                            unary_vizlib_feats[span.stable_id].add((f, v))

                    for f, v in unary_vizlib_feats[span.stable_id]:
                        yield candidate.id, FEAT_PRE + pre + f, v

                if candidate.id not in binary_vizlib_feats:
                    binary_vizlib_feats[candidate.id] = set()
                    for f, v in vizlib_binary_features(span1, span2):
                        binary_vizlib_feats[candidate.id].add((f, v))

                for f, v in binary_vizlib_feats[candidate.id]:
                    yield candidate.id, FEAT_PRE + f, v
        else:
            raise NotImplementedError("Only handles unary and binary candidates currently")


def vizlib_unary_features(span):
    """
    Visual-related features for a single span
    """
    if not span.sentence.is_visual(): return

    for f in get_visual_aligned_lemmas(span):
        yield 'ALIGNED_' + f, DEF_VALUE

    for page in set(span.get_attrib_tokens('page')):
        yield "PAGE_[%d]" % page, DEF_VALUE


def vizlib_binary_features(span1, span2):
    """
    Visual-related features for a pair of spans
    """
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