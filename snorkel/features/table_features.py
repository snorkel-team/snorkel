from snorkel.lf_helpers import *
from snorkel.models import ImplicitSpan
from snorkel.table_utils import min_row_diff, min_col_diff, num_rows, num_cols

FEAT_PRE = 'TAB_'
DEF_VALUE = 1


def get_table_feats(candidate):
    args = candidate.get_arguments()
    if not (isinstance(args[0], TemporarySpan)):
        raise ValueError("Accepts Span-type arguments, %s-type found." % type(candidate))

    # Unary candidates
    if len(args) == 1:
        get_tablelib_feats = tablelib_unary_features
        span = args[0]
        # Add TableLib entity features (if applicable)
        if isinstance(span.parent, Phrase):
            # if span.has_table_features():
            for f, v in get_tablelib_feats(span):
                yield FEAT_PRE + f, v
    # Binary candidates
    elif len(args) == 2:
        get_tablelib_feats = tablelib_binary_features
        span1, span2 = args
        # Add TableLib relation features (if applicable)
        if isinstance(span1.parent, Phrase) or isinstance(span2.parent, Phrase):
            # if span1.has_table_features() or span2.has_table_features():
            for f, v in get_tablelib_feats(span1, span2):
                yield FEAT_PRE + f, v
    else:
        raise NotImplementedError("Only handles unary and binary candidates currently")


def tablelib_unary_features(span):
    """
    Table-/structure-related features for a single span
    """
    yield "SPAN_TYPE_[%s]" % ('IMPLICIT' if isinstance(span, ImplicitSpan) else 'EXPLICIT'), 1
    phrase = span.parent
    if phrase.html_tag:
        yield u"HTML_TAG_" + phrase.html_tag, DEF_VALUE
    # Comment out for now, we could calc it later.
    # for attr in phrase.html_attrs:
    #     yield u"HTML_ATTR_[" + attr + "]", DEF_V
    # if phrase.html_anc_tags:
    #     for tag in phrase.html_anc_tags:
    #         yield u"HTML_ANC_TAG_[" + tag + "]", DEF_VALUE
            # for attr in phrase.html_anc_attrs:
            # yield u"HTML_ANC_ATTR_[" + attr + "]"
    for attrib in ['words']:  # ,'lemmas', 'pos_tags', 'ner_tags']:
        for ngram in span.get_attrib_tokens(attrib):
            yield "CONTAINS_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
        for ngram in get_left_ngrams(span, window=7, n_max=2, attrib=attrib):
            yield "LEFT_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
        for ngram in get_right_ngrams(span, window=7, n_max=2, attrib=attrib):
            yield "RIGHT_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
        if phrase.row_start is None or phrase.col_start is None:
            for ngram in get_neighbor_phrase_ngrams(span, d=1, n_max=2, attrib=attrib):
                yield "NEIGHBOR_PHRASE_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
        else:
            for ngram in get_cell_ngrams(span, n_max=2, attrib=attrib):
                yield "CELL_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
            for row_num in range(phrase.row_start, phrase.row_end + 1):
                yield "ROW_NUM_[%s]" % row_num, DEF_VALUE
            for col_num in range(phrase.col_start, phrase.col_end + 1):
                yield "COL_NUM_[%s]" % col_num, DEF_VALUE
            # NOTE: These two features should be accounted for by HTML_ATTR_
            yield "ROW_SPAN_[%d]" % num_rows(phrase), DEF_VALUE
            yield "COL_SPAN_[%d]" % num_cols(phrase), DEF_VALUE
            for axis in ['row', 'col']:
                for ngram in get_head_ngrams(span, axis, n_max=2, attrib=attrib):
                    yield "%s_HEAD_%s_[%s]" % (axis.upper(), attrib.upper(), ngram), 1
            for ngram in get_row_ngrams(span, n_max=2, attrib=attrib):
                yield "ROW_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
            for ngram in get_col_ngrams(span, n_max=2, attrib=attrib):
                yield "COL_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
            # for ngram in get_row_ngrams(span, n_max=2, attrib=attrib, direct=False, infer=True):
            #     yield "ROW_INFERRED_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
            # for ngram in get_col_ngrams(span, n_max=2, attrib=attrib, direct=False, infer=True):
            #     yield "COL_INFERRED_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
                # for (ngram, direction) in get_neighbor_cell_ngrams(span, dist=2, directions=True, n_max=3, \
                #  attrib=attrib):
                #     yield "NEIGHBOR_%s_%s_[%s]" % (direction, attrib.upper(), ngram)
                #     if attrib=="lemmas":
                #         try:
                #             if float(ngram).is_integer():
                #                 yield "NEIGHBOR_%s_INT" % side
                #             else:
                #                 yield "NEIGHBOR_%s_FLOAT" % side
                #         except:
                #             pass


def tablelib_binary_features(span1, span2):
    """
    Table-/structure-related features for a pair of spans
    """
    for feat, v in tablelib_unary_features(span1):
        yield "e1_" + feat, v
    for feat, v in tablelib_unary_features(span2):
        yield "e2_" + feat, v
    if span1.parent.table is not None and span2.parent.table is not None:
        if span1.parent.table == span2.parent.table:
            yield u"SAME_TABLE", DEF_VALUE
            if span1.parent.cell is not None and span2.parent.cell is not None:
                row_diff = min_row_diff(span1.parent, span2.parent, absolute=False)
                col_diff = min_col_diff(span1.parent, span2.parent, absolute=False)
                yield u"SAME_TABLE_ROW_DIFF_[%s]" % row_diff, DEF_VALUE
                yield u"SAME_TABLE_COL_DIFF_[%s]" % col_diff, DEF_VALUE
                yield u"SAME_TABLE_MANHATTAN_DIST_[%s]" % str(abs(row_diff) + abs(col_diff)), DEF_VALUE
                if span1.parent.cell == span2.parent.cell:
                    yield u"SAME_CELL", DEF_VALUE
                    yield u"WORD_DIFF_[%s]" % (span1.get_word_start() - span2.get_word_start()), DEF_VALUE
                    yield u"CHAR_DIFF_[%s]" % (span1.char_start - span2.char_start), DEF_VALUE
                    if span1.parent == span2.parent:
                        yield u"SAME_PHRASE", DEF_VALUE
        else:
            if span1.parent.cell is not None and span2.parent.cell is not None:
                row_diff = min_row_diff(span1.parent, span2.parent, absolute=False)
                col_diff = min_col_diff(span1.parent, span2.parent, absolute=False)
                yield u"DIFF_TABLE_ROW_DIFF_[%s]" % row_diff, DEF_VALUE
                yield u"DIFF_TABLE_COL_DIFF_[%s]" % col_diff, DEF_VALUE
                yield u"DIFF_TABLE_MANHATTAN_DIST_[%s]" % str(abs(row_diff) + abs(col_diff)), DEF_VALUE
