from __future__ import unicode_literals
from builtins import str
from builtins import range
from ..lf_helpers import *
from ..utils_table import min_row_diff, min_col_diff, num_rows, num_cols
from ..config import settings

FEAT_PRE = 'TAB_'
DEF_VALUE = 1

unary_tablelib_feats = {}
binary_tablelib_feats = {}


def get_table_feats(candidates):
    candidates = candidates if isinstance(candidates, list) else [candidates]
    for candidate in candidates:
        args = candidate.get_contexts()
        if not (isinstance(args[0], TemporarySpan)):
            raise ValueError("Accepts Span-type arguments, %s-type found." % type(candidate))

        # Unary candidates
        if len(args) == 1:
            span = args[0]
            if span.stable_id not in unary_tablelib_feats:
                unary_tablelib_feats[span.stable_id] = set()
                for f, v in tablelib_unary_features(span):
                    unary_tablelib_feats[span.stable_id].add((f, v))

            for f, v in unary_tablelib_feats[span.stable_id]:
                yield candidate.id, FEAT_PRE + f, v

        # Binary candidates
        elif len(args) == 2:
            span1, span2 = args
            if span1.sentence.is_tabular() or span2.sentence.is_tabular():
                for span, pre in [(span1, "e1_"), (span2, "e2_")]:
                    if span.stable_id not in unary_tablelib_feats:
                        unary_tablelib_feats[span.stable_id] = set()
                        for f, v in tablelib_unary_features(span):
                            unary_tablelib_feats[span.stable_id].add((f, v))

                    for f, v in unary_tablelib_feats[span.stable_id]:
                        yield candidate.id, FEAT_PRE + pre + f, v

                if candidate.id not in binary_tablelib_feats:
                    binary_tablelib_feats[candidate.id] = set()
                    for f, v in tablelib_binary_features(span1, span2):
                        binary_tablelib_feats[candidate.id].add((f, v))

                for f, v in binary_tablelib_feats[candidate.id]:
                    yield candidate.id, FEAT_PRE + f, v
        else:
            raise NotImplementedError("Only handles unary and binary candidates currently")


def tablelib_unary_features(span):
    """
    Table-/structure-related features for a single span
    """
    if not span.sentence.is_tabular(): return
    phrase = span.sentence
    for attrib in settings.featurization.table.unary_features.attrib:
        for ngram in get_cell_ngrams(span,
                                     n_max=settings.featurization.table.unary_features.get_cell_ngrams.max,
                                     attrib=attrib):
            yield "CELL_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
        for row_num in range(phrase.row_start, phrase.row_end + 1):
            yield "ROW_NUM_[%s]" % row_num, DEF_VALUE
        for col_num in range(phrase.col_start, phrase.col_end + 1):
            yield "COL_NUM_[%s]" % col_num, DEF_VALUE
        # NOTE: These two features could be accounted for by HTML_ATTR in structural features
        yield "ROW_SPAN_[%d]" % num_rows(phrase), DEF_VALUE
        yield "COL_SPAN_[%d]" % num_cols(phrase), DEF_VALUE
        for axis in ['row', 'col']:
            for ngram in get_head_ngrams(span, axis,
                                         n_max=settings.featurization.table.unary_features.get_head_ngrams.max,
                                         attrib=attrib):
                yield "%s_HEAD_%s_[%s]" % (axis.upper(), attrib.upper(), ngram), 1
        for ngram in get_row_ngrams(span, n_max=settings.featurization.table.unary_features.get_row_ngrams.max,
                                    attrib=attrib):
            yield "ROW_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
        for ngram in get_col_ngrams(span, n_max=settings.featurization.table.unary_features.get_col_ngrams.max,
                                    attrib=attrib):
            yield "COL_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
            # TODO:
            # for ngram in get_row_ngrams(span, n_max=2, attrib=attrib, direct=False, infer=True):
            #     yield "ROW_INFERRED_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
            # for ngram in get_col_ngrams(span, n_max=2, attrib=attrib, direct=False, infer=True):
            #     yield "COL_INFERRED_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE


def tablelib_binary_features(span1, span2):
    """
    Table-/structure-related features for a pair of spans
    """
    if span1.sentence.is_tabular() and span2.sentence.is_tabular():
        if span1.sentence.table == span2.sentence.table:
            yield u"SAME_TABLE", DEF_VALUE
            if span1.sentence.cell is not None and span2.sentence.cell is not None:
                row_diff = min_row_diff(span1.sentence, span2.sentence,
                                        absolute=settings.featurization.table.binary_features.min_row_diff.absolute)
                col_diff = min_col_diff(span1.sentence, span2.sentence,
                                        absolute=settings.featurization.table.binary_features.min_col_diff.absolute)
                yield u"SAME_TABLE_ROW_DIFF_[%s]" % row_diff, DEF_VALUE
                yield u"SAME_TABLE_COL_DIFF_[%s]" % col_diff, DEF_VALUE
                yield u"SAME_TABLE_MANHATTAN_DIST_[%s]" % str(abs(row_diff) + abs(col_diff)), DEF_VALUE
                if span1.sentence.cell == span2.sentence.cell:
                    yield u"SAME_CELL", DEF_VALUE
                    yield u"WORD_DIFF_[%s]" % (span1.get_word_start() - span2.get_word_start()), DEF_VALUE
                    yield u"CHAR_DIFF_[%s]" % (span1.char_start - span2.char_start), DEF_VALUE
                    if span1.sentence == span2.sentence:
                        yield u"SAME_PHRASE", DEF_VALUE
        else:
            if span1.sentence.cell is not None and span2.sentence.cell is not None:
                yield u"DIFF_TABLE", DEF_VALUE
                row_diff = min_row_diff(span1.sentence, span2.sentence,
                                        absolute=settings.featurization.table.binary_features.min_row_diff.absolute)
                col_diff = min_col_diff(span1.sentence, span2.sentence,
                                        absolute=settings.featurization.table.binary_features.min_col_diff.absolute)
                yield u"DIFF_TABLE_ROW_DIFF_[%s]" % row_diff, DEF_VALUE
                yield u"DIFF_TABLE_COL_DIFF_[%s]" % col_diff, DEF_VALUE
                yield u"DIFF_TABLE_MANHATTAN_DIST_[%s]" % str(abs(row_diff) + abs(col_diff)), DEF_VALUE
