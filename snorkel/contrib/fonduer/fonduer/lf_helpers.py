from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
from collections import defaultdict
from itertools import chain
from lxml import etree
from lxml.html import fromstring
import numpy as np

from snorkel.utils import tokens_to_ngrams
from .utils_visual import bbox_from_span, bbox_from_phrase, bbox_horz_aligned, bbox_vert_aligned, bbox_vert_aligned_left, bbox_vert_aligned_right, bbox_vert_aligned_center
from .utils_table import min_row_diff, min_col_diff, is_row_aligned, is_col_aligned, is_axis_aligned
from ....models.context import TemporarySpan
from .models import Phrase
from snorkel.candidates import Ngrams


def get_between_ngrams(c, attrib='words', n_min=1, n_max=1, lower=True):
    """Return the ngrams _between_ two unary Spans of a binary-Span Candidate.

    Get the ngrams _between_ two unary Spans of a binary-Span Candidate, where
    both share the same sentence Context.

    :param c: The binary-Span Candidate to evaluate.
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If 'True', all ngrams will be returned in lower case
    :rtype: a _generator_ of ngrams
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


def get_left_ngrams(span, window=3, attrib='words', n_min=1, n_max=1, lower=True):
    """Get the ngrams within a window to the _left_ of the Candidate from its sentence Context.

    For higher-arity Candidates, defaults to the _first_ argument.

    :param span: The Span to evaluate. If a candidate is given, default to its first Span.
    :param window: The number of tokens to the left of the first argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a _generator_ of ngrams
    """
    span = span if isinstance(
        span, TemporarySpan) else span[0]  # get first Span
    i = span.get_word_start()
    for ngram in tokens_to_ngrams(getattr(span.sentence, attrib)[max(0, i - window):i],
                                  n_min=n_min, n_max=n_max,
                                  lower=lower):
        yield ngram


def get_right_ngrams(span, window=3, attrib='words', n_min=1, n_max=1, lower=True):
    """Get the ngrams within a window to the _right_ of the Candidate from its sentence Context.

    For higher-arity Candidates, defaults to the _last_ argument.

    :param span: The Span to evaluate. If a candidate is given, default to its last Span.
    :param window: The number of tokens to the left of the first argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a _generator_ of ngrams
    """
    span = span if isinstance(
        span, TemporarySpan) else span[-1]  # get last Span
    i = span.get_word_end()
    for ngram in tokens_to_ngrams(getattr(span.sentence, attrib)[i + 1:i + 1 + window],
                                  n_min=n_min, n_max=n_max,
                                  lower=lower):
        yield ngram


def get_matches(lf, candidate_set, match_values=[1, -1]):
    """Return a list of candidates that are matched by a particular LF.

    A simple helper function to see how many matches (non-zero by default) an LF gets.
    Returns the matched candidates, which can then be directly put into the Viewer.

    :param lf: The labeling function to apply to the candidate_set
    :param candidate_set: The set of candidates to evaluate
    :param match_values: An option list of the values to consider as matched. [1, -1] by default.
    :rtype: a list of candidates
    """
    matches = []
    for c in candidate_set:
        label = lf(c)
        if label in match_values:
            matches.append(c)
    print(("%s matches") % len(matches))
    return matches


# TABLE LF HELPERS ##########################################################
def same_document(c):
    """Return True if all Spans in the given candidate are from the same Document.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return (all(c[i].sentence.document is not None
                and c[i].sentence.document == c[0].sentence.document for i in range(len(c))))


def same_table(c):
    """Return True if all Spans in the given candidate are from the same Table.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return (all(c[i].sentence.is_tabular() and
                c[i].sentence.table == c[0].sentence.table for i in range(len(c))))


def same_row(c):
    """Return True if all Spans in the given candidate are from the same Row.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return (same_table(c) and
            all(is_row_aligned(c[i].sentence, c[0].sentence)
                for i in range(len(c))))


def same_col(c):
    """Return True if all Spans in the given candidate are from the same Col.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return (same_table(c) and
            all(is_col_aligned(c[i].sentence, c[0].sentence)
                for i in range(len(c))))


def is_tabular_aligned(c):
    """Return True if all Spans in the given candidate are from the same Row or Col.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return (same_table(c) and
            (is_col_aligned(c[i].sentence, c[0].sentence) or
             is_row_aligned(c[i].sentence, c[0].sentence)
             for i in range(len(c))))


def same_cell(c):
    """Return True if all Spans in the given candidate are from the same Cell.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return (all(c[i].sentence.cell is not None and
                c[i].sentence.cell == c[0].sentence.cell for i in range(len(c))))


def same_phrase(c):
    """Return True if all Spans in the given candidate are from the same Phrase.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return (all(c[i].sentence is not None
                and c[i].sentence == c[0].sentence for i in range(len(c))))


def get_max_col_num(span):
    """Return the largest column number that a Span occupies.

    :param span: The Span to evaluate. If a candidate is given, default to its last Span.
    :rtype: integer or None
    """
    span = span if isinstance(span, TemporarySpan) else span[-1]
    if span.sentence.is_tabular():
        return span.sentence.cell.col_end
    else:
        return None


def get_min_col_num(span):
    """Return the lowest column number that a Span occupies.

    :param span: The Span to evaluate. If a candidate is given, default to its first Span.
    :rtype: integer or None
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    if span.sentence.is_tabular():
        return span.sentence.cell.col_start
    else:
        return None


def get_phrase_ngrams(span, attrib='words', n_min=1, n_max=1, lower=True):
    """Get the ngrams that are in the Phrase of the given span, not including itself.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The Span whose Phrase is being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a _generator_ of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in get_left_ngrams(span, window=100, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram
        for ngram in get_right_ngrams(span, window=100, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram


def get_neighbor_phrase_ngrams(span, d=1, attrib='words', n_min=1, n_max=1, lower=True):
    """Get the ngrams that are in the neighoring Phrases of the given Span.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The span whose neighbor Phrases are being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a _generator_ of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in chain.from_iterable(
                [tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower)
                 for phrase in span.sentence.document.phrases
                 if abs(phrase.phrase_num - span.sentence.phrase_num) <= d and phrase != span.sentence]):
            yield ngram


def get_cell_ngrams(span, attrib='words', n_min=1, n_max=1, lower=True):
    """Get the ngrams that are in the Cell of the given span, not including itself.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The span whose Cell is being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a _generator_ of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in get_phrase_ngrams(span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram
        if isinstance(span.sentence, Phrase) and span.sentence.cell is not None:
            for ngram in chain.from_iterable(
                    [tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower)
                     for phrase in span.sentence.cell.phrases if phrase != span.sentence]):
                yield ngram


def get_neighbor_cell_ngrams(span, dist=1, directions=False, attrib='words', n_min=1, n_max=1, lower=True):
    """Get the ngrams from all Cells that are within a given Cell distance in one direction from the given Span.

    Note that if a candidate is passed in, all of its Spans will be searched.
    If `directions=True``, each ngram will be returned with a direction in {'UP', 'DOWN', 'LEFT', 'RIGHT'}.

    :param span: The span whose neighbor Cells are being searched
    :param dist: The Cell distance within which a neighbor Cell must be to be considered
    :param directions: A Boolean expressing whether or not to return the direction of each ngram
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a _generator_ of ngrams (or (ngram, direction) tuples if directions=True)
    """
    # TODO: Fix this to be more efficient (optimize with SQL query)
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in get_phrase_ngrams(span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram
        if isinstance(span.sentence, Phrase) and span.sentence.cell is not None:
            root_cell = span.sentence.cell
            for phrase in chain.from_iterable([_get_aligned_phrases(root_cell, 'row'), _get_aligned_phrases(root_cell, 'col')]):
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


def get_row_ngrams(span, attrib='words', n_min=1, n_max=1, spread=[0, 0], lower=True):
    """Get the ngrams from all Cells that are in the same row as the given Span.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The span whose row Cells are being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a _generator_ of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in _get_axis_ngrams(span, axis='row', attrib=attrib, 
                                      n_min=n_min, n_max=n_max, spread=spread, lower=lower):
            yield ngram


def get_col_ngrams(span, attrib='words', n_min=1, n_max=1, spread=[0, 0], lower=True):
    """Get the ngrams from all Cells that are in the same column as the given Span.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The span whose column Cells are being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a _generator_ of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in _get_axis_ngrams(span, axis='col', attrib=attrib, 
                                      n_min=n_min, n_max=n_max, spread=spread, 
                                      lower=lower):
            yield ngram


def get_aligned_ngrams(span, attrib='words', n_min=1, n_max=1, spread=[0, 0], lower=True):
    """Get the ngrams from all Cells that are in the same row or column as the given Span.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The span whose row and column Cells are being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a _generator_ of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in get_row_ngrams(span, attrib=attrib, n_min=n_min, 
                                    n_max=n_max, spread=spread, lower=lower):
            yield ngram
        for ngram in get_col_ngrams(span, attrib=attrib, n_min=n_min, 
                                    n_max=n_max, spread=spread, lower=lower):
            yield ngram


def get_head_ngrams(span, axis=None, attrib='words', n_min=1, n_max=1, lower=True):
    """Get the ngrams from the cell in the head of the row or column.

    More specifically, this returns the ngrams in the leftmost cell in a row and/or the
    ngrams in the topmost cell in the column, depending on the axis parameter.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The span whose head Cells are being returned
    :param axis: Which axis {'row', 'col'} to search. If None, then both row and col are searched.
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a _generator_ of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    axes = [axis] if axis else ['row', 'col']
    for span in spans:
        if not span.sentence.cell:
            return
        else:
            for axis in axes:
                if getattr(span.sentence, _other_axis(axis) + '_start') == 0:
                    return
                for phrase in getattr(_get_head_cell(span.sentence.cell, axis), 'phrases', []):
                    for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                        yield ngram


def _get_head_cell(root_cell, axis):
    other_axis = 'row' if axis == 'col' else 'col'
    aligned_cells = _get_aligned_cells(
        root_cell, axis)
    return sorted(aligned_cells, key=lambda x: getattr(x, other_axis + '_start'))[0] if aligned_cells else []


def _get_axis_ngrams(span, axis, attrib='words', n_min=1, n_max=1, spread=[0, 0], lower=True):
    for ngram in get_phrase_ngrams(span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
        yield ngram
    if (span.sentence.cell is not None):
        for phrase in _get_aligned_phrases(span.sentence, axis, spread=spread):
            for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                yield ngram


def _get_aligned_cells(root_cell, axis):
    aligned_cells = [cell for cell in root_cell.table.cells
                     if is_axis_aligned(root_cell, cell, axis=axis)
                     and cell != root_cell]
    return aligned_cells


def _get_aligned_phrases(root_phrase, axis, spread=[0, 0]):
    return [phrase for cell in root_phrase.table.cells if is_axis_aligned(root_phrase, cell, axis=axis, spread=spread)
            for phrase in cell.phrases if phrase != root_phrase]


def _other_axis(axis):
    return 'row' if axis == 'col' else 'col'


def is_superset(a, b):
    """Check if a is a superset of b.

    This is typically used to check if ALL of a list of phrases is in the ngrams returned by an lf_helper.

    :param a: A collection of items
    :param b: A collection of items
    :rtype: boolean
    """
    return set(a).issuperset(b)


def overlap(a, b):
    """Check if a overlaps b.

    This is typically used to check if ANY of a list of phrases is in the ngrams returned by an lf_helper.

    :param a: A collection of items
    :param b: A collection of items
    :rtype: boolean
    """
    return not set(a).isdisjoint(b)


############################
# Visual feature helpers
############################
def get_page(span):
    """Return the page number of the given span.

    If a candidate is passed in, this returns the page of its first Span.

    :param span: The Span to get the page number of.
    :rtype: integer
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    return span.get_attrib_tokens('page')[0]


def is_horz_aligned(c):
    """Return True if all the components of c are horizontally aligned.

    Horizontal alignment means that the bounding boxes of each Span of c shares
    a similar y-axis value in the visual rendering of the document.

    :param c: The candidate to evaluate
    :rtype: boolean
    """
    return (all([c[i].sentence.is_visual() and
                 bbox_horz_aligned(bbox_from_span(c[i]), bbox_from_span(c[0]))
                 for i in range(len(c))]))


def is_vert_aligned(c):
    """Return true if all the components of c are vertically aligned.

    Vertical alignment means that the bounding boxes of each Span of c shares
    a similar x-axis value in the visual rendering of the document.

    :param c: The candidate to evaluate
    :rtype: boolean
    """
    return (all([c[i].sentence.is_visual() and
                 bbox_vert_aligned(bbox_from_span(c[i]), bbox_from_span(c[0]))
                 for i in range(len(c))]))


def is_vert_aligned_left(c):
    """Return true if all the components of c are vertically aligned based on their left border.

    Vertical alignment means that the bounding boxes of each Span of c shares
    a similar x-axis value in the visual rendering of the document. In this function
    the similarity of the x-axis value is based on the left border of their bounding boxes.

    :param c: The candidate to evaluate
    :rtype: boolean
    """
    return (all([c[i].sentence.is_visual() and
                 bbox_vert_aligned_left(
                     bbox_from_span(c[i]), bbox_from_span(c[0]))
                 for i in range(len(c))]))


def is_vert_aligned_right(c):
    """Return true if all the components of c are vertically aligned based on their right border.

    Vertical alignment means that the bounding boxes of each Span of c shares
    a similar x-axis value in the visual rendering of the document. In this function
    the similarity of the x-axis value is based on the right border of their bounding boxes.

    :param c: The candidate to evaluate
    :rtype: boolean
    """
    return (all([c[i].sentence.is_visual() and
                 bbox_vert_aligned_right(
                     bbox_from_span(c[i]), bbox_from_span(c[0]))
                 for i in range(len(c))]))


def is_vert_aligned_center(c):
    """Return true if all the components of c are vertically aligned based on their left border.

    Vertical alignment means that the bounding boxes of each Span of c shares
    a similar x-axis value in the visual rendering of the document. In this function
    the similarity of the x-axis value is based on the center of their bounding boxes.

    :param c: The candidate to evaluate
    :rtype: boolean
    """
    return (all([c[i].sentence.is_visual() and
                 bbox_vert_aligned_center(
                     bbox_from_span(c[i]), bbox_from_span(c[0]))
                 for i in range(len(c))]))


def same_page(c):
    """Return true if all the components of c are on the same page of the document.

    Page numbers are based on the PDF rendering of the document. If a PDF file is
    provided, it is used. Otherwise, if only a HTML/XML document is provided, a
    PDF is created and then used to determine the page number of a Span.

    :param c: The candidate to evaluate
    :rtype: boolean
    """
    return (all([c[i].sentence.is_visual() and
                 bbox_from_span(c[i]).page == bbox_from_span(c[0]).page
                 for i in range(len(c))]))


def get_horz_ngrams(span, attrib='words', n_min=1, n_max=1, lower=True, from_phrase=True):
    """Return all ngrams which are visually horizontally aligned with the Span.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The Span to evaluate
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :param from_phrase: If True, returns ngrams from any horizontally aligned Phrases,
                        rather than just horizontally aligned ngrams themselves.
    :rtype: a _generator_ of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in _get_direction_ngrams('horz', span, attrib, n_min, n_max, lower, from_phrase):
            yield ngram


def get_vert_ngrams(span, attrib='words', n_min=1, n_max=1, lower=True, from_phrase=True):
    """Return all ngrams which are visually vertivally aligned with the Span.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The Span to evaluate
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :param from_phrase: If True, returns ngrams from any horizontally aligned Phrases,
                        rather than just horizontally aligned ngrams themselves.
    :rtype: a _generator_ of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in _get_direction_ngrams('vert', span, attrib, n_min, n_max, lower, from_phrase):
            yield ngram


def _get_direction_ngrams(direction, c, attrib, n_min, n_max, lower, from_phrase):
    # TODO: this currently looks only in current table;
    #   precompute over the whole document/page instead
    bbox_direction_aligned = bbox_vert_aligned if direction == 'vert' else bbox_horz_aligned
    ngrams_space = Ngrams(n_max=n_max, split_tokens=[])
    f = (lambda w: w.lower()) if lower else (lambda w: w)
    spans = [c] if isinstance(c, TemporarySpan) else c.get_contexts()
    for span in spans:
        if not span.sentence.is_tabular() or not span.sentence.is_visual():
            continue
        for phrase in span.sentence.table.phrases:
            if (from_phrase):
                if (bbox_direction_aligned(bbox_from_phrase(phrase), bbox_from_span(span)) and
                        phrase is not span.sentence):
                    for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                        yield ngram
            else:
                for ts in ngrams_space.apply(phrase):
                    if (bbox_direction_aligned(bbox_from_span(ts), bbox_from_span(span)) and
                            not (phrase == span.sentence and ts.get_span() in span.get_span())):
                        yield f(ts.get_span())


def get_vert_ngrams_left(c):
    """Not implemented."""
    # TODO
    return


def get_vert_ngrams_right(c):
    """Not implemented."""
    # TODO
    return


def get_vert_ngrams_center(c):
    """Not implemented."""
    # TODO
    return


def get_visual_header_ngrams(c, axis=None):
    """Not implemented."""
    # TODO
    return


def get_visual_distance(c, axis=None):
    """Not implemented."""
    # TODO
    return


# Default dimensions for 8.5" x 11"
DEFAULT_WIDTH = 612
DEFAULT_HEIGHT = 792


def get_page_vert_percentile(span, page_width=DEFAULT_WIDTH, page_height=DEFAULT_HEIGHT):
    """Return which percentile from the TOP in the page Span candidate is located in.

    Percentile is calculated where the top of the page is 0.0, and the bottom of
    the page is 1.0. For example, a Span in at the top 1/4 of the page will have
    a percentil of 0.25.

    Page width and height are based on pt values:
        Letter      612x792
        Tabloid     792x1224
        Ledger      1224x792
        Legal       612x1008
        Statement   396x612
        Executive   540x720
        A0          2384x3371
        A1          1685x2384
        A2          1190x1684
        A3          842x1190
        A4          595x842
        A4Small     595x842
        A5          420x595
        B4          729x1032
        B5          516x729
        Folio       612x936
        Quarto      610x780
        10x14       720x1008
    and should match the source documents. Letter size is used by default.

    Note that if a candidate is passed in, only the vertical percentil of its
    first Span is returned.

    :param span: The Span to evaluate
    :param page_width: The width of the page. Default to Letter paper width.
    :param page_height: The heigh of the page. Default to Letter paper height.
    :rtype: float in [0.0, 1.0]
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    return old_div(float(bbox_from_span(span).top), page_height)


def get_page_horz_percentile(span, page_width=DEFAULT_WIDTH, page_height=DEFAULT_HEIGHT):
    """Return which percentile from the LEFT in the page the Span is located in.

    Percentile is calculated where the left of the page is 0.0, and the right of
    the page is 1.0.

    Page width and height are based on pt values:
        Letter      612x792
        Tabloid     792x1224
        Ledger      1224x792
        Legal       612x1008
        Statement   396x612
        Executive   540x720
        A0          2384x3371
        A1          1685x2384
        A2          1190x1684
        A3          842x1190
        A4          595x842
        A4Small     595x842
        A5          420x595
        B4          729x1032
        B5          516x729
        Folio       612x936
        Quarto      610x780
        10x14       720x1008
    and should match the source documents. Letter size is used by default.

    Note that if a candidate is passed in, only the vertical percentil of its
    first Span is returned.

    :param c: The Span to evaluate
    :param page_width: The width of the page. Default to Letter paper width.
    :param page_height: The heigh of the page. Default to Letter paper height.
    :rtype: float in [0.0, 1.0]
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    return old_div(float(bbox_from_span(span).left), page_width)


def _assign_alignment_features(phrases_by_key, align_type):
    for key, phrases in phrases_by_key.items():
        if len(phrases) == 1:
            continue
        context_lemmas = set()
        for p in phrases:
            p._aligned_lemmas.update(context_lemmas)
            # update lemma context for upcoming phrases in the group
            if len(p.lemmas) < 7:
                new_lemmas = [lemma.lower()
                              for lemma in p.lemmas if lemma.isalpha()]
                # if new_lemmas: print '++Lemmas for\t', p, context_lemmas
                context_lemmas.update(new_lemmas)
                context_lemmas.update(
                    align_type + lemma for lemma in new_lemmas)


def _preprocess_visual_features(doc):
    if hasattr(doc, '_visual_features'):
        return
    # cache flag
    doc._visual_features = True

    phrase_by_page = defaultdict(list)
    for phrase in doc.phrases:
        phrase_by_page[phrase.page[0]].append(phrase)
        phrase._aligned_lemmas = set()

    for page, phrases in phrase_by_page.items():
        # process per page alignments
        yc_aligned = defaultdict(list)
        x0_aligned = defaultdict(list)
        xc_aligned = defaultdict(list)
        x1_aligned = defaultdict(list)
        for phrase in phrases:
            phrase.bbox = bbox_from_phrase(phrase)
            phrase.yc = old_div((phrase.bbox.top + phrase.bbox.bottom), 2)
            phrase.x0 = phrase.bbox.left
            phrase.x1 = phrase.bbox.right
            phrase.xc = old_div((phrase.x0 + phrase.x1), 2)
            # index current phrase by different alignment keys
            yc_aligned[phrase.yc].append(phrase)
            x0_aligned[phrase.x0].append(phrase)
            x1_aligned[phrase.x1].append(phrase)
            xc_aligned[phrase.xc].append(phrase)
        for l in yc_aligned.values():
            l.sort(key=lambda p: p.xc)
        for l in x0_aligned.values():
            l.sort(key=lambda p: p.yc)
        for l in x1_aligned.values():
            l.sort(key=lambda p: p.yc)
        for l in xc_aligned.values():
            l.sort(key=lambda p: p.yc)
        _assign_alignment_features(yc_aligned, 'Y_')
        _assign_alignment_features(x0_aligned, 'LEFT_')
        _assign_alignment_features(x1_aligned, 'RIGHT_')
        _assign_alignment_features(xc_aligned, 'CENTER_')


def get_visual_aligned_lemmas(span):
    """Return a generator of the lemmas aligned visually with the Span.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The Span to evaluate.
    :rtype: a _generator_ of lemmas
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        phrase = span.sentence
        doc = phrase.document
        # cache features for the entire document
        _preprocess_visual_features(doc)

        for aligned_lemma in phrase._aligned_lemmas:
            yield aligned_lemma


def get_aligned_lemmas(span):
    """Return a set of the lemmas aligned visually with the Span.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The Span to evaluate.
    :rtype: a set of lemmas
    """
    return set(get_visual_aligned_lemmas(span))


############################
# Structural feature helpers
############################
def get_tag(span):
    """Return the HTML tag of the Span.

    If a candidate is passed in, only the tag of its first Span is returned.

    These may be tags such as 'p', 'h2', 'table', 'div', etc.
    :param span: The Span to evaluate
    :rtype: string
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    return str(span.sentence.html_tag)


def get_attributes(span):
    """Return the HTML attributes of the Span.

    If a candidate is passed in, only the tag of its first Span is returned.

    A sample outout of this function on a Span in a paragraph tag is
    [u'style=padding-top: 8pt;padding-left: 20pt;text-indent: 0pt;text-align: left;']

    :param span: The Span to evaluate
    :rtype: list of strings representing HTML attributes
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    return span.sentence.html_attrs


# TODO: Too slow
def _get_node(phrase):
    return (etree.ElementTree(fromstring(phrase.document.text)).xpath(phrase.xpath))[0]


def get_parent_tag(span):
    """Return the HTML tag of the Span's parent.

    These may be tags such as 'p', 'h2', 'table', 'div', etc.
    If a candidate is passed in, only the tag of its first Span is returned.

    :param span: The Span to evaluate
    :rtype: string
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    i = _get_node(span.sentence)
    return str(i.getparent().tag) if i.getparent() is not None else None


def get_prev_sibling_tags(span):
    """Return the HTML tag of the Span's previous siblings.

    Previous siblings are Spans which are at the same level in the HTML tree as
    the given span, but are declared before the given span.
    If a candidate is passed in, only the previous siblings of its first Span
    are considered in the calculation.

    :param span: The Span to evaluate
    :rtype: list of strings
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    prev_sibling_tags = []
    i = _get_node(span.sentence)
    while i.getprevious() is not None:
        prev_sibling_tags.insert(0, str(i.getprevious().tag))
        i = i.getprevious()
    return prev_sibling_tags


def get_next_sibling_tags(span):
    """Return the HTML tag of the Span's next siblings.

    Next siblings are Spans which are at the same level in the HTML tree as
    the given span, but are declared after the given span.
    If a candidate is passed in, only the next siblings of its last Span
    are considered in the calculation.

    :param span: The Span to evaluate
    :rtype: list of strings
    """
    span = span if isinstance(span, TemporarySpan) else span[-1]
    next_sibling_tags = []
    i = _get_node(span.sentence)
    while i.getnext() is not None:
        next_sibling_tags.append(str(i.getnext().tag))
        i = i.getnext()
    return next_sibling_tags


def get_ancestor_class_names(span):
    """Return the HTML classes of the Span's ancestors.

    If a candidate is passed in, only the ancestors of its first Span are returned.

    :param span: The Span to evaluate
    :rtype: list of strings
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    class_names = []
    i = _get_node(span.sentence)
    while i is not None:
        class_names.insert(0, str(i.get('class')))
        i = i.getparent()
    return class_names


def get_ancestor_tag_names(span):
    """Return the HTML tag of the Span's ancestors.

    For example, ['html', 'body', 'p'].
    If a candidate is passed in, only the ancestors of its first Span are returned.

    :param span: The Span to evaluate
    :rtype: list of strings
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    tag_names = []
    i = _get_node(span.sentence)
    while i is not None:
        tag_names.insert(0, str(i.tag))
        i = i.getparent()
    return tag_names


def get_ancestor_id_names(span):
    """Return the HTML id's of the Span's ancestors.

    If a candidate is passed in, only the ancestors of its first Span are returned.

    :param span: The Span to evaluate
    :rtype: list of strings
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    id_names = []
    i = _get_node(span.sentence)
    while i is not None:
        id_names.insert(0, str(i.get('id')))
        i = i.getparent()
    return id_names


def common_ancestor(c):
    """Return the common path to the root that is shared between a binary-Span Candidate.

    In particular, this is the common path of HTML tags.

    :param c: The binary-Span Candidate to evaluate
    :rtype: list of strings
    """
    ancestor1 = np.array(c[0].sentence.xpath.split('/'))
    ancestor2 = np.array(c[1].sentence.xpath.split('/'))
    min_len = min(ancestor1.size, ancestor2.size)
    return list(ancestor1[:np.argmin(ancestor1[:min_len] == ancestor2[:min_len])])


def lowest_common_ancestor_depth(c):
    """Return the minimum distance between a binary-Span Candidate to their lowest common ancestor.

    For example, if the tree looked like this:

        html
        |----<div> span 1 </div>
        |----table
        |    |----tr
        |    |    |-----<th> span 2 </th>

    we return 1, the distance from span 1 to the html root. Smaller values indicate
    that two Spans are close structurally, while larger values indicate that two
    Spans are spread far apart structurally in the document.

    :param c: The binary-Span Candidate to evaluate
    :rtype: integer
    """
    ancestor1 = np.array(c[0].sentence.xpath.split('/'))
    ancestor2 = np.array(c[1].sentence.xpath.split('/'))
    min_len = min(ancestor1.size, ancestor2.size)
    return min_len - np.argmin(ancestor1[:min_len] == ancestor2[:min_len])
