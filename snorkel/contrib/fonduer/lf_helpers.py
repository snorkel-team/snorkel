import re
from collections import defaultdict
from itertools import chain
from lxml.html import fromstring
from lxml import etree
import numpy as np

from snorkel.utils import tokens_to_ngrams
from utils_table import *
from utils_visual import *
from snorkel.models import TemporarySpan, Phrase
from snorkel.candidates import Ngrams


def get_text_splits(c):
    """
    Given a k-arity Candidate defined over k Spans, return the chunked sentence context (e.g. Sentence)
    split around the k constituent Spans.

    NOTE: Currently assumes that these Spans are in the same Context
    """
    spans = []
    for i, span in enumerate(c.get_arguments()):
        if not isinstance(span, TemporarySpan):
            raise ValueError("Handles Span-type Candidate arguments only")

        # Note: {{0}}, {{1}}, etc. does not work as an un-escaped regex pattern, hence A, B, ...
        spans.append((span.char_start, span.char_end, chr(65 + i)))
    spans.sort()

    # NOTE: Assume all Spans in same sentence Context
    text = span.sentence.text

    # Get text chunks
    chunks = [text[:spans[0][0]], "{{%s}}" % spans[0][2]]
    for j in range(len(spans) - 1):
        chunks.append(text[spans[j][1] + 1:spans[j + 1][0]])
        chunks.append("{{%s}}" % spans[j + 1][2])
    chunks.append(text[spans[-1][1] + 1:])
    return chunks


def get_tagged_text(c):
    """
    Returns the text of c's sentence context with c's unary spans replaced with
    tags {{A}}, {{B}}, etc.
    A convenience method for writing LFs based on e.g. regexes.
    """
    return "".join(get_text_splits(c))


def get_text_between(c):
    """
    Returns the text between the two unary Spans of a binary-Span Candidate, where
    both are in the same Sentence.
    """
    chunks = get_text_splits(c)
    if len(chunks) == 5:
        return chunks[2]
    else:
        raise ValueError("Only applicable to binary Candidates")


# TODO: replace in tutorials with get_between_ngrams and delete this
def get_between_tokens(c, attrib='words', n_min=1, n_max=1, lower=True):
    """ An alias for get_between_ngrams maintained for backwards compatibility. """
    return [ngram for ngram in get_between_ngrams(c,
                                                  attrib=attrib,
                                                  n_min=n_min,
                                                  n_max=n_max,
                                                  lower=lower)]


def get_between_ngrams(c, attrib='words', n_min=1, n_max=1, lower=True):
    """
    Get the ngrams _between_ two unary Spans of a binary-Span Candidate, where
    both share the same sentence Context.
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If true, all ngrams will be returned in lower case
    """
    if len(c) != 2:
        raise ValueError("Only applicable to binary Candidates")
    span0 = c[0]
    span1 = c[1]
    if span0.sentence != span1.sentence:
        raise ValueError("Only applicable to Candidates where both spans are \
                          from the same immediate Context.")
    distance = abs(span0.get_word_start() - span1.get_word_start())
    if span0.get_word_start() < span1.get_word_start():
        for ngram in get_right_ngrams(span0,
                                      window=distance - 1, attrib=attrib,
                                      n_min=n_min, n_max=n_max,
                                      lower=lower):
            yield ngram
    else:  # span0.get_word_start() > span1.get_word_start()
        for ngram in get_left_ngrams(span1,
                                     window=distance - 1, attrib=attrib,
                                     n_min=n_min, n_max=n_max,
                                     lower=lower):
            yield ngram


# TODO: replace in tutorials with get_left_ngrams and delete this
def get_left_tokens(c, window=3, attrib='words', n_min=1, n_max=1, lower=True):
    """ An alias for get_left_ngrams maintained for backwards compatibility. """
    return [ngram for ngram in get_left_ngrams(c,
                                               window=window, attrib=attrib,
                                               n_min=n_min, n_max=n_max,
                                               lower=lower)]


def get_left_ngrams(c, window=3, attrib='words', n_min=1, n_max=1, lower=True):
    """
    Get the ngrams within a window to the _left_ of the Candidate from its sentence Context.
    For higher-arity Candidates, defaults to the _first_ argument.
    :param window: The number of tokens to the left of the first argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If true, all ngrams will be returned in lower case
    """
    span = c if isinstance(c, TemporarySpan) else c[0]
    i = span.get_word_start()
    for ngram in tokens_to_ngrams(getattr(span.sentence, attrib)[max(0, i - window):i],
                                  n_min=n_min, n_max=n_max,
                                  lower=lower):
        yield ngram


# TODO: replace in tutorials with get_right_ngrams and delete this
def get_right_tokens(c, window=3, attrib='words', n_min=1, n_max=1, lower=True):
    """ An alias for get_right_ngrams maintained for backwards compatibility. """
    return [ngram for ngram in get_right_ngrams(c, window=window, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower)]


def get_right_ngrams(c, window=3, attrib='words', n_min=1, n_max=1, lower=True):
    """
    Get the ngrams within a window to the _right_ of the Candidate from its sentence Context.
    For higher-arity Candidates, defaults to the _last_ argument.
    :param window: The number of tokens to the left of the first argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If true, all ngrams will be returned in lower case
    """
    span = c if isinstance(c, TemporarySpan) else c[-1]
    i = span.get_word_end()
    for ngram in tokens_to_ngrams(getattr(span.sentence, attrib)[i + 1:i + 1 + window], n_min=n_min, n_max=n_max,
                                  lower=lower):
        yield ngram


def contains_token(c, tok, attrib='words', lower=True):
    """
    Return True if any of the contituent Spans contain the given token
    :param tok: The token being searched for
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param lower: If true, all ngrams will be returned in lower case
    """
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    f = (lambda w: w.lower()) if lower else (lambda w: w)
    return f(tok) in set(chain.from_iterable(map(f, span.get_attrib_tokens(attrib))
                                             for span in spans))


def contains_regex(c, rgx=None, attrib='words', sep=" ", ignore_case=True):
    """
    Return True if any of the contituent Spans contain the given regular expression
    :param rgx: The regex being searched for
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param sep: The separator to be used in concatening the retrieved tokens
    :param lower: If true, all ngrams will be returned in lower case
    """
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    r = re.compile(rgx, flags=re.I if ignore_case else 0)
    return any([r.search(span.get_attrib_span(attrib, sep=sep)) is not None for span in spans])


def get_doc_candidate_spans(c):
    """
    Get the Spans in the same document as Candidate c, where these Spans are
    arguments of Candidates.
    """
    # TODO: Fix this to be more efficient and properly general!!
    spans = list(chain.from_iterable(s.spans for s in c[0].sentence.document.sentences))
    return [s for s in spans if s != c[0]]


def get_sent_candidate_spans(c):
    """
    Get the Spans in the same Sentence as Candidate c, where these Spans are
    arguments of Candidates.
    """
    # TODO: Fix this to be more efficient and properly general!!
    return [s for s in c[0].sentence.spans if s != c[0]]


def get_matches(lf, candidate_set, match_values=[1, -1]):
    """
    A simple helper function to see how many matches (non-zero by default) an LF gets.
    Returns the matched set, which can then be directly put into the Viewer.
    """
    matches = []
    for c in candidate_set:
        label = lf(c)
        if label in match_values:
            matches.append(c)
    print "%s matches" % len(matches)
    return matches


############################### TABLE LF HELPERS ###############################
def same_document(c):
    """
    Return True if all Spans in the given candidate are from the same Document.
    :param c: The candidate whose Spans are being compared
    """
    return (all(c[i].sentence.document is not None
                and c[i].sentence.document == c[0].sentence.document for i in range(len(c))))


def same_table(c):
    """
    Return True if all Spans in the given candidate are from the same Table.
    :param c: The candidate whose Spans are being compared
    """
    return (all(c[i].is_tabular() and
                c[i].sentence.table == c[0].sentence.table for i in range(len(c))))


def same_row(c):
    """
    Return True if all Spans in the given candidate are from the same Row.
    :param c: The candidate whose Spans are being compared
    """
    return (same_table(c) and
            all(is_row_aligned(c[i].sentence, c[0].sentence)
                for i in range(len(c))))


def same_col(c):
    """
    Return True if all Spans in the given candidate are from the same Col.
    :param c: The candidate whose Spans are being compared
    """
    return (same_table(c) and
            all(is_col_aligned(c[i].sentence, c[0].sentence)
                for i in range(len(c))))


def is_tabular_aligned(c):
    """
    Return True if all Spans in the given candidate are from the same Row
    or Col
    :param c: The candidate whose Spans are being compared
    """
    return (same_table(c) and
            (is_col_aligned(c[i].sentence, c[0].sentence) or
             is_row_aligned(c[i].sentence, c[0].sentence)
             for i in range(len(c))))


def same_cell(c):
    """
    Return True if all Spans in the given candidate are from the same Cell.
    :param c: The candidate whose Spans are being compared
    """
    return (all(c[i].sentence.cell is not None and
                c[i].sentence.cell == c[0].sentence.cell for i in range(len(c))))


def same_phrase(c):
    """
    Return True if all Spans in the given candidate are from the same Phrase.
    :param c: The candidate whose Spans are being compared
    """
    return (all(c[i].sentence is not None
                and c[i].sentence == c[0].sentence for i in range(len(c))))


def get_max_col_num(c):
    span = c if isinstance(c, TemporarySpan) else c.get_arguments()[0]
    if span.is_tabular():
        return span.sentence.cell.col_end
    else:
        return None


def get_min_col_num(c):
    span = c if isinstance(c, TemporarySpan) else c.get_arguments()[0]
    if span.is_tabular():
        return span.sentence.cell.col_start
    else:
        return None


def get_phrase_ngrams(c, attrib='words', n_min=1, n_max=1, lower=True):
    """
    Get the ngrams that are in the Phrase of the given span, not including itself.
    :param span: The span whose Phrase is being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If true, all ngrams will be returned in lower case
    """
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        for ngram in get_left_ngrams(span, window=100, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram
        for ngram in get_right_ngrams(span, window=100, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram


def get_neighbor_phrase_ngrams(c, d=1, attrib='words', n_min=1, n_max=1, lower=True):
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        for ngram in chain.from_iterable(
                [tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower)
                 for phrase in span.sentence.document.phrases
                 if abs(phrase.phrase_num - span.sentence.phrase_num) <= d and phrase != span.sentence]):
            yield ngram


def get_cell_ngrams(c, attrib='words', n_min=1, n_max=1, lower=True):
    """
    Get the ngrams that are in the Cell of the given span, not including itself.
    :param span: The span whose Cell is being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If true, all ngrams will be returned in lower case
    """
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        for ngram in get_phrase_ngrams(span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram
        if isinstance(span.sentence, Phrase) and span.sentence.cell is not None:
            for ngram in chain.from_iterable(
                    [tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower) \
                     for phrase in span.sentence.cell.phrases if phrase != span.sentence]):
                yield ngram


def get_neighbor_cell_ngrams(c, dist=1, directions=False, attrib='words', n_min=1, n_max=1, lower=True):
    """
    Get the ngrams from all Cells that are within a given Cell distance in one direction from the given span
    :param span: The span whose neighbor Cells are being searched
    :param dist: The Cell distance within which a neighbor Cell must be to be considered
    :param directions: A Boolean expressing whether or not to return the direction of each ngram
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If true, all ngrams will be returned in lower case
    """
    # TODO: Fix this to be more efficient (optimize with SQL query)
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        for ngram in get_phrase_ngrams(span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram
        if isinstance(span.sentence, Phrase) and span.sentence.cell is not None:
            root_cell = span.sentence.cell
            for phrase in chain.from_iterable(
                    [_get_aligned_phrases(phrase, 'row'), _get_aligned_phrases(phrase, 'col')]):
                row_diff = min_row_diff(phrase, root_cell, absolute=False)
                col_diff = min_col_diff(phrase, root_cell, absolute=False)
                if (row_diff or col_diff) and not (row_diff and col_diff) and abs(row_diff) + abs(col_diff) <= dist:
                    if directions:
                        direction = ''
                        if col_diff == 0:
                            if 0 < row_diff and row_diff <= dist:
                                direction = "UP"
                            elif 0 > row_diff and row_diff >= -dist:
                                direction = "DOWN"
                        elif row_diff == 0:
                            if 0 < col_diff and col_diff <= dist:
                                direction = "RIGHT"
                            elif 0 > col_diff and col_diff >= -dist:
                                direction = "LEFT"
                        for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                            yield (ngram, direction)
                    else:
                        for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                            yield ngram


def get_row_ngrams(c, direct=True, infer=False, attrib='words', n_min=1, n_max=1, spread=[0, 0], lower=True):
    """
    Get the ngrams from all Cells that are in the same row as the given span
    :param span: The span whose neighbor Cells are being searched
    :param infer: If True, then if a Cell is empty, use the contents from the first non-empty Cell above it
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If true, all ngrams will be returned in lower case
    """
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        for ngram in _get_axis_ngrams(span, axis='row', direct=direct, infer=infer,
                                      attrib=attrib, n_min=n_min, n_max=n_max, spread=spread, lower=lower):
            yield ngram


def get_col_ngrams(c, direct=True, infer=False, attrib='words', n_min=1, n_max=1, spread=[0, 0], lower=True):
    """
    Get the ngrams from all Cells that are in the same column as the given span
    :param span: The span whose neighbor Cells are being searched
    :param infer: If True, then if a Cell is empty, use the contents from the first non-empty Cell to the left of it
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If true, all ngrams will be returned in lower case
    """
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        for ngram in _get_axis_ngrams(span, axis='col', direct=direct, infer=infer,
                                      attrib=attrib, n_min=n_min, n_max=n_max, spread=spread, lower=lower):
            yield ngram


def get_aligned_ngrams(c, direct=True, infer=False, attrib='words', n_min=1, n_max=1, spread=[0, 0], lower=True):
    """
    Get the ngrams from all Cells that are in the same row or column as the given span
    :param span: The span whose neighbor Cells are being searched
    :param infer: Refer to get_[row/col]_ngrams for description
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If true, all ngrams will be returned in lower case
    """
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        for ngram in get_row_ngrams(span, direct=direct, infer=infer, attrib=attrib,
                                    n_min=n_min, n_max=n_max, spread=spread, lower=lower):
            yield ngram
        for ngram in get_col_ngrams(span, direct=direct, infer=infer, attrib=attrib,
                                    n_min=n_min, n_max=n_max, spread=spread, lower=lower):
            yield ngram


def get_head_ngrams(c, axis=None, infer=False, attrib='words', n_min=1, n_max=1, lower=True):
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    axes = [axis] if axis else ['row', 'col']
    for span in spans:
        if not span.sentence.cell:
            return
        else:
            for axis in axes:
                if getattr(span.sentence, _other_axis(axis) + '_start') == 0: return
                for phrase in getattr(_get_head_cell(span.sentence.cell, axis, infer=infer), 'phrases', []):
                    for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                        yield ngram


def _get_head_cell(root_cell, axis, infer=False):
    other_axis = 'row' if axis == 'col' else 'col'
    aligned_cells = _get_aligned_cells(root_cell, axis, direct=True, infer=infer)
    return sorted(aligned_cells, key=lambda x: getattr(x, other_axis + '_start'))[0] if aligned_cells else []


def _get_axis_ngrams(span, axis, direct=True, infer=False, attrib='words', n_min=1, n_max=1, spread=[0, 0], lower=True):
    for ngram in get_phrase_ngrams(span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
        yield ngram
    if (span.sentence.cell is not None):
        for phrase in _get_aligned_phrases(span.sentence, axis, direct=direct, infer=infer, spread=spread):
            for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                yield ngram


def _get_aligned_cells(root_cell, axis, direct=True, infer=False):
    aligned_cells = [cell for cell in root_cell.table.cells
                     if is_axis_aligned(root_cell, cell, axis=axis)
                     and cell != root_cell]
    return [_infer_cell(cell, _other_axis(axis), direct=direct, infer=infer) \
            for cell in aligned_cells] if infer else aligned_cells


def _get_aligned_phrases(root_phrase, axis, direct=True, infer=False, spread=[0, 0]):
    return [phrase for cell in root_phrase.table.cells if is_axis_aligned(root_phrase, cell, axis=axis, spread=spread) \
            for phrase in _infer_cell(cell, _other_axis(axis), direct, infer).phrases \
            if phrase != root_phrase]


# PhantomCell = namedtuple('PhantomCell','phrases')
# TODO: fix this function and retest
def _infer_cell(root_cell, axis, direct, infer):
    # NOTE: not defined for direct = False and infer = False
    # empty = len(root_cell.phrases) == 0 
    # edge = getattr(root_cell, _other_axis(axis) + '_start') == 0
    # if direct and (not empty or edge or not infer):
    #     return root_cell
    # else:
    #     if edge or not empty:
    #         return PhantomCell(phrases=[]) 
    #     else:
    #         neighbor_cells = [cell for cell in root_cell.table.cells
    #             if is_axis_aligned(cell, root_cell, axis=axis)
    #             and getattr(cell, _other_axis(axis) + '_start') == \
    #                 getattr(root_cell, _other_axis(axis) + '_start') - 1]
    #         return _infer_cell(neighbor_cells[0], axis, direct=True, infer=True)
    return root_cell


def _other_axis(axis):
    return 'row' if axis == 'col' else 'col'


def is_superset(a, b):
    return set(a).issuperset(b)


def overlap(a, b):
    return not set(a).isdisjoint(b)


############################
# Visual feature helpers
############################
def get_page(c):
    span = c if isinstance(c, TemporarySpan) else c.get_arguments()[0]
    return span.get_attrib_tokens('page')[0]


def is_horz_aligned(c):
    return (all([c[i].is_visual() and
                 bbox_horz_aligned(bbox_from_span(c[i]), bbox_from_span(c[0]))
                 for i in range(len(c))]))


def is_vert_aligned(c):
    return (all([c[i].is_visual() and
                 bbox_vert_aligned(bbox_from_span(c[i]), bbox_from_span(c[0]))
                 for i in range(len(c))]))


def is_vert_aligned_left(c):
    return (all([c[i].is_visual() and
                 bbox_vert_aligned_left(bbox_from_span(c[i]), bbox_from_span(c[0]))
                 for i in range(len(c))]))


def is_vert_aligned_right(c):
    return (all([c[i].is_visual() and
                 bbox_vert_aligned_right(bbox_from_span(c[i]), bbox_from_span(c[0]))
                 for i in range(len(c))]))


def is_vert_aligned_center(c):
    return (all([c[i].is_visual() and
                 bbox_vert_aligned_center(bbox_from_span(c[i]), bbox_from_span(c[0]))
                 for i in range(len(c))]))


def same_page(c):
    return (all([c[i].is_visual() and
                 bbox_from_span(c[i]).page == bbox_from_span(c[0]).page
                 for i in range(len(c))]))


def get_horz_ngrams(c, attrib='words', n_min=1, n_max=1, lower=True):
    for ngram in _get_direction_ngrams('horz', c, attrib, n_min, n_max, lower):
        yield ngram


def get_vert_ngrams(c, attrib='words', n_min=1, n_max=1, lower=True):
    for ngram in _get_direction_ngrams('vert', c, attrib, n_min, n_max, lower):
        yield ngram


def _get_direction_ngrams(direction, c, attrib, n_min, n_max, lower):
    # TODO: this currently looks only in current table;
    #   precompute over the whole document/page instead
    bbox_direction_aligned = bbox_vert_aligned if direction == 'vert' else bbox_horz_aligned
    ngrams_space = Ngrams(n_max=n_max, split_tokens=[])
    f = (lambda w: w.lower()) if lower else (lambda w: w)
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        if not span.is_tabular() or not span.is_visual(): continue
        for phrase in span.sentence.table.phrases:
            for ts in ngrams_space.apply(phrase):
                if (bbox_direction_aligned(bbox_from_span(ts), bbox_from_span(span)) and
                        not (phrase == span.sentence and ts.get_span() in span.get_span())):
                    yield ts.get_span()


def get_vert_ngrams_left(c):
    # TODO
    return


def get_vert_ngrams_right(c):
    # TODO
    return


def get_vert_ngrams_center(c):
    # TODO
    return


def get_visual_header_ngrams(c, axis=None):
    # TODO
    return


def get_visual_distance(c, axis=None):
    # TODO
    return


# Default dimensions for 8.5" x 11"
DEFAULT_WIDTH = 612
DEFAULT_HEIGHT = 792


def get_page_vert_percentile(c, page_width=DEFAULT_WIDTH, page_height=DEFAULT_HEIGHT):
    span = c if isinstance(c, TemporarySpan) else c.get_arguments()[0]
    return float(bbox_from_span(span).top) / page_height


def get_page_horz_percentile(c, page_width=DEFAULT_WIDTH, page_height=DEFAULT_HEIGHT):
    span = c if isinstance(c, TemporarySpan) else c.get_arguments()[0]
    return float(bbox_from_span(span).left) / page_width


def _assign_alignment_features(phrases_by_key, align_type):
    for key, phrases in phrases_by_key.iteritems():
        if len(phrases) == 1: continue
        context_lemmas = set()
        for p in phrases:
            p._aligned_lemmas.update(context_lemmas)
            # update lemma context for upcoming phrases in the group
            if len(p.lemmas) < 7:
                new_lemmas = [lemma.lower() for lemma in p.lemmas if lemma.isalpha()]
                #                 if new_lemmas: print '++Lemmas for\t', p, context_lemmas
                context_lemmas.update(new_lemmas)
                context_lemmas.update(align_type + lemma for lemma in new_lemmas)


def _preprocess_visual_features(doc):
    if hasattr(doc, '_visual_features'): return
    # cache flag
    doc._visual_features = True

    phrase_by_page = defaultdict(list)
    for phrase in doc.phrases:
        phrase_by_page[phrase.page[0]].append(phrase)
        phrase._aligned_lemmas = set()

    for page, phrases in phrase_by_page.iteritems():
        # process per page alignments
        yc_aligned = defaultdict(list)
        x0_aligned = defaultdict(list)
        xc_aligned = defaultdict(list)
        x1_aligned = defaultdict(list)
        for phrase in phrases:
            phrase.bbox = bbox_from_phrase(phrase)
            phrase.yc = (phrase.bbox.top + phrase.bbox.bottom) / 2
            phrase.x0 = phrase.bbox.left
            phrase.x1 = phrase.bbox.right
            phrase.xc = (phrase.x0 + phrase.x1) / 2
            # index current phrase by different alignment keys
            yc_aligned[phrase.yc].append(phrase)
            x0_aligned[phrase.x0].append(phrase)
            x1_aligned[phrase.x1].append(phrase)
            xc_aligned[phrase.xc].append(phrase)
        for l in yc_aligned.itervalues(): l.sort(key=lambda p: p.yc)
        for l in x0_aligned.itervalues(): l.sort(key=lambda p: p.x0)
        for l in x1_aligned.itervalues(): l.sort(key=lambda p: p.x1)
        for l in xc_aligned.itervalues(): l.sort(key=lambda p: p.xc)
        _assign_alignment_features(yc_aligned, 'Y_')
        _assign_alignment_features(x0_aligned, 'LEFT_')
        _assign_alignment_features(x1_aligned, 'RIGHT_')
        _assign_alignment_features(xc_aligned, 'CENTER_')


def get_visual_aligned_lemmas(span):
    phrase = span.sentence
    doc = phrase.document
    # cache features for the entire document
    _preprocess_visual_features(doc)

    for aligned_lemma in phrase._aligned_lemmas:
        yield aligned_lemma


def get_aligned_lemmas(span):
    return set(get_visual_aligned_lemmas(span))


############################
# Structural feature helpers
############################
def get_tag(span):
    return str(span.sentence.html_tag)


def get_attributes(span):
    return span.sentence.html_attrs


# TODO: Too slow
def _get_node(phrase):
    return (etree.ElementTree(fromstring(phrase.document.text)).xpath(phrase.xpath))[0]


def get_parent_tag(span):
    i = _get_node(span.sentence)
    return str(i.getparent().tag) if i.getparent() is not None else None


def get_prev_sibling_tags(span):
    prev_sibling_tags = []
    i = _get_node(span.sentence)
    while i.getprevious() is not None:
        prev_sibling_tags.insert(0, str(i.getprevious().tag))
        i = i.getprevious()
    return prev_sibling_tags


def get_next_sibling_tags(span):
    next_sibling_tags = []
    i = _get_node(span.sentence)
    while i.getnext() is not None:
        next_sibling_tags.append(str(i.getnext().tag))
        i = i.getnext()
    return next_sibling_tags


def get_ancestor_class_names(span):
    class_names = []
    i = _get_node(span.sentence)
    while i is not None:
        class_names.insert(0, str(i.get('class')))
        i = i.getparent()
    return class_names


def get_ancestor_tag_names(span):
    tag_names = []
    i = _get_node(span.sentence)
    while i is not None:
        tag_names.insert(0, str(i.tag))
        i = i.getparent()
    return tag_names


def get_ancestor_id_names(span):
    id_names = []
    i = _get_node(span.sentence)
    while i is not None:
        id_names.insert(0, str(i.get('id')))
        i = i.getparent()
    return id_names


def common_ancestor(c):
    ancestor1 = np.array(c[0].sentence.xpath.split('/'))
    ancestor2 = np.array(c[1].sentence.xpath.split('/'))
    min_len = min(ancestor1.size, ancestor2.size)
    return ancestor1[:np.argmin(ancestor1[:min_len] == ancestor2[:min_len])]


def lowest_common_ancestor_depth(c):
    ancestor1 = np.array(c[0].sentence.xpath.split('/'))
    ancestor2 = np.array(c[1].sentence.xpath.split('/'))
    min_len = min(ancestor1.size, ancestor2.size)
    return min_len - np.argmin(ancestor1[:min_len] == ancestor2[:min_len])
