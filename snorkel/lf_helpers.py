import re
from collections import namedtuple, defaultdict
from itertools import chain

from table_utils import min_row_diff, min_col_diff, is_axis_aligned, is_row_aligned, is_col_aligned
from utils import tokens_to_ngrams
from .models import TemporarySpan, Phrase


def get_text_splits(c):
    """
    Given a k-arity Candidate defined over k Spans, return the chunked parent context (e.g. Sentence)
    split around the k constituent Spans.

    NOTE: Currently assumes that these Spans are in the same Context
    """
    spans = []
    for i, span in enumerate(c.get_arguments()):
        if not isinstance(span, TemporarySpan):
            raise ValueError("Handles Span-type Candidate arguments only")

        # Note: {{0}}, {{1}}, etc. does not work as an un-escaped regex pattern, hence A, B, ...
        spans.append((span.char_start, span.char_end, chr(65+i)))
    spans.sort()

    # NOTE: Assume all Spans in same parent Context
    text = span.parent.text

    # Get text chunks
    chunks = [text[:spans[0][0]], "{{%s}}" % spans[0][2]]
    for j in range(len(spans)-1):
        chunks.append(text[spans[j][1]+1:spans[j+1][0]])
        chunks.append("{{%s}}" % spans[j+1][2])
    chunks.append(text[spans[-1][1]+1:])
    return chunks


def get_tagged_text(c):
    """
    Returns the text of c's parent context with c's unary spans replaced with
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
    both share the same parent Context.
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If false, all ngrams will be returned in lower case
    """
    if len(c) != 2:
        raise ValueError("Only applicable to binary Candidates")
    span0 = c[0]
    span1 = c[1]
    if span0.parent != span1.parent:
        raise ValueError("Only applicable to Candidates where both spans are \
                          from the same immediate Context.")
    distance = abs(span0.get_word_start() - span1.get_word_start())
    if span0.get_word_start() < span1.get_word_start():
        for ngram in get_right_ngrams(span0,
                                      window=distance-1, attrib=attrib,
                                      n_min=n_min, n_max=n_max,
                                      lower=lower):
            yield ngram
    else: # span0.get_word_start() > span1.get_word_start()
        for ngram in get_left_ngrams(span1,
                                     window=distance-1, attrib=attrib,
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
    Get the ngrams within a window to the _left_ of the Candidate from its parent Context.
    For higher-arity Candidates, defaults to the _first_ argument.
    :param window: The number of tokens to the left of the first argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If false, all ngrams will be returned in lower case
    """
    span = c if isinstance(c, TemporarySpan) else c[0]
    i = span.get_word_start()
    for ngram in tokens_to_ngrams(getattr(span.parent, attrib)[max(0, i-window):i],
                                  n_min=n_min, n_max=n_max,
                                  lower=lower):
        yield ngram


# TODO: replace in tutorials with get_right_ngrams and delete this
def get_right_tokens(c, window=3, attrib='words', n_min=1, n_max=1, lower=True):
    """ An alias for get_right_ngrams maintained for backwards compatibility. """
    return [ngram for ngram in get_right_ngrams(c, window=window, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower)]


def get_right_ngrams(c, window=3, attrib='words', n_min=1, n_max=1, lower=True):
    """
    Get the ngrams within a window to the _right_ of the Candidate from its parent Context.
    For higher-arity Candidates, defaults to the _last_ argument.
    :param window: The number of tokens to the left of the first argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If false, all ngrams will be returned in lower case
    """
    span = c if isinstance(c, TemporarySpan) else c[-1]
    i    = span.get_word_end()
    for ngram in tokens_to_ngrams(getattr(span.parent, attrib)[i+1:i+1+window], n_min=n_min, n_max=n_max, lower=lower):
        yield ngram


def contains_token(c, tok, attrib='words', lower=True):
    """
    Return True if any of the contituent Spans contain the given token
    :param tok: The token being searched for
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param lower: If false, all ngrams will be returned in lower case
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
    :param lower: If false, all ngrams will be returned in lower case
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
    spans = list(chain.from_iterable(s.spans for s in c[0].parent.document.sentences))
    return [s for s in spans if s != c[0]]


def get_sent_candidate_spans(c):
    """
    Get the Spans in the same Sentence as Candidate c, where these Spans are
    arguments of Candidates.
    """
    # TODO: Fix this to be more efficient and properly general!!
    return [s for s in c[0].parent.spans if s != c[0]]


def get_matches(lf, candidate_set, match_values=[1,-1]):
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
    return (all(c[i].parent.document is not None
        and c[i].parent.document == c[0].parent.document for i in range(len(c))))


def same_table(c):
    """
    Return True if all Spans in the given candidate are from the same Table.
    :param c: The candidate whose Spans are being compared
    """
    return (all(isinstance(c[i], TablePhrase) and
                c[i].parent.table == c[0].parent.table for i in range(len(c))))


def same_row(c):
    """
    Return True if all Spans in the given candidate are from the same Row.
    :param c: The candidate whose Spans are being compared
    """
    return (all(isinstance(c[i], TablePhrase) and 
                c[i].parent.table == c[0].parent.table and 
                is_row_aligned(c[i].parent, c[0].parent)
            for i in range(len(c))))


def same_col(c):
    """
    Return True if all Spans in the given candidate are from the same Col.
    :param c: The candidate whose Spans are being compared
    """
    return (all(isinstance(c[i], TablePhrase) and 
                c[i].parent.table == c[0].parent.table and 
                is_col_aligned(c[i].parent, c[0].parent)
            for i in range(len(c))))
    

def is_tabular_aligned(c):
    """
    Return True if all Spans in the given candidate are from the same Row
    or Col
    :param c: The candidate whose Spans are being compared
    """
    return same_table(c) and (is_col_aligned(c[i].parent, c[0].parent) or
                              is_row_aligned(c[i].parent, c[0].parent))


def same_cell(c):
    """
    Return True if all Spans in the given candidate are from the same Cell.
    :param c: The candidate whose Spans are being compared
    """
    return (all(isinstance(c[i], TablePhrase) and
                c[i].parent.cell is not None and
                c[i].parent.cell == c[0].parent.cell for i in range(len(c))))


def same_phrase(c):
    """
    Return True if all Spans in the given candidate are from the same Phrase.
    :param c: The candidate whose Spans are being compared
    """
    return (all(c[i].parent is not None
        and c[i].parent==c[0].parent for i in range(len(c))))


def get_phrase_ngrams(c, attrib='words', n_min=1, n_max=1, lower=True):
    """
    Get the ngrams that are in the Phrase of the given span, not including itself.
    :param span: The span whose Phrase is being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If false, all ngrams will be returned in lower case
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
            for phrase in span.parent.document.phrases 
            if abs(phrase.position - span.parent.position) <= d and phrase != span.parent]):
            yield ngram


def get_cell_ngrams(c, attrib='words', n_min=1, n_max=1, lower=True):
    """
    Get the ngrams that are in the Cell of the given span, not including itself.
    :param span: The span whose Cell is being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If false, all ngrams will be returned in lower case
    """
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        for ngram in get_phrase_ngrams(span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram
        if isinstance(span.parent, Phrase) and span.parent.cell is not None:
            for ngram in chain.from_iterable([tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower) \
                for phrase in span.parent.cell.phrases if phrase != span.parent]):
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
    :param lower: If false, all ngrams will be returned in lower case
    """
    # TODO: Fix this to be more efficient (optimize with SQL query)
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        for ngram in get_phrase_ngrams(span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram
        if isinstance(span.parent, TablePhrase) and span.parent.cell is not None:
            root_cell = span.parent.cell
            for phrase in chain.from_iterable([_get_aligned_phrases(phrase, 'row'), _get_aligned_phrases(phrase, 'col')]):
                row_diff = min_row_diff(phrase, root_cell, absolute=False) 
                col_diff = min_col_diff(phrase, root_cell, absolute=False) 
                if (row_diff or col_diff) and not (row_diff and col_diff) and abs(row_diff) + abs(col_diff) <= dist:
                    if directions:
                        direction = ''
                        if col_diff==0:
                            if 0 < row_diff and row_diff <= dist:
                                direction = "UP"
                            elif  0 > row_diff and row_diff >= -dist:
                                direction = "DOWN"
                        elif row_diff==0:
                            if 0 < col_diff and col_diff <= dist:
                                direction = "RIGHT"
                            elif  0 > col_diff and col_diff >= -dist:
                                direction = "LEFT"
                        for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                            yield (ngram, direction)
                    else:
                        for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                            yield  ngram


def get_row_ngrams(c, direct=True, infer=False, attrib='words', n_min=1, n_max=1, spread=[0,0], lower=True):
    """
    Get the ngrams from all Cells that are in the same row as the given span
    :param span: The span whose neighbor Cells are being searched
    :param infer: If True, then if a Cell is empty, use the contents from the first non-empty Cell above it
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If false, all ngrams will be returned in lower case
    """
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        for ngram in _get_axis_ngrams(span, axis='row', direct=direct, infer=infer, 
            attrib=attrib, n_min=n_min, n_max=n_max, spread=spread, lower=lower):
            yield ngram


def get_col_ngrams(c, direct=True, infer=False, attrib='words', n_min=1, n_max=1, spread=[0,0], lower=True):
    """
    Get the ngrams from all Cells that are in the same column as the given span
    :param span: The span whose neighbor Cells are being searched
    :param infer: If True, then if a Cell is empty, use the contents from the first non-empty Cell to the left of it
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If false, all ngrams will be returned in lower case
    """
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        for ngram in _get_axis_ngrams(span, axis='col', direct=direct, infer=infer, 
        attrib=attrib, n_min=n_min, n_max=n_max, spread=spread, lower=lower):
            yield ngram


def get_aligned_ngrams(c, direct=True, infer=False, attrib='words', n_min=1, n_max=1, spread=[0,0], lower=True):
    """
    Get the ngrams from all Cells that are in the same row or column as the given span
    :param span: The span whose neighbor Cells are being searched
    :param infer: Refer to get_[row/col]_ngrams for description
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If false, all ngrams will be returned in lower case
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
    axes = [axis] if axis else ['row','col']
    for span in spans:
        if not span.parent.cell:
            return
        else:
            for axis in axes:
                for phrase in getattr(_get_head_cell(span.parent.cell, axis, infer=infer), 'phrases', []):
                    for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                        yield ngram


def _get_head_cell(root_cell, axis, infer=False):
    other_axis = 'row' if axis=='col' else 'col'
    aligned_cells = _get_aligned_cells(root_cell, axis, direct=True, infer=infer)  
    return sorted(aligned_cells, key=lambda x: getattr(x, other_axis + '_start'))[0] if aligned_cells else []


def _get_axis_ngrams(span, axis, direct=True, infer=False, attrib='words', n_min=1, n_max=1, spread=[0,0], lower=True):
    for ngram in get_phrase_ngrams(span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
        yield ngram
    if (span.parent.cell is not None):
        for phrase in _get_aligned_phrases(span.parent, axis, direct=direct, infer=infer, spread=spread):
            for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                yield ngram


def _get_aligned_cells(root_cell, axis, direct=True, infer=False):
    aligned_cells = [cell for cell in root_cell.table.cells
        if is_axis_aligned(root_cell, cell, axis=axis)
        and cell != root_cell]
    return [_infer_cell(cell, _other_axis(axis), direct=direct, infer=infer) \
        for cell in aligned_cells] if infer else aligned_cells


def _get_aligned_phrases(root_phrase, axis, direct=True, infer=False, spread=[0,0]):
    return [phrase for cell in root_phrase.table.cells if is_axis_aligned(root_phrase, cell, axis=axis, spread=spread)\
                for phrase in _infer_cell(cell, _other_axis(axis), direct, infer).phrases \
                    if phrase!=root_phrase]


PhantomCell = namedtuple('PhantomCell','phrases')
def _infer_cell(root_cell, axis, direct, infer):
    # NOTE: not defined for direct = False and infer = False
    # TODO: Fix this hack; checking for len(text)==9 checks if cell is "<td></td>"
    empty = len(root_cell.text) == 9
    edge = getattr(root_cell, _other_axis(axis) + '_start') == 0
    if direct and (not empty or edge or not infer):
        return root_cell
    else:
        if edge or not empty:
            return PhantomCell(phrases=[]) 
        else:
            neighbor_cells = [cell for cell in root_cell.table.cells
                if getattr(cell, axis + '_start') == getattr(root_cell, axis + '_start')
                and getattr(cell, _other_axis(axis) + '_start').position == \
                    getattr(root_cell, _other_axis(axis) + '_start').position - 1]
            return _infer_cell(neighbor_cells[0], axis, direct=True, infer=True)

def _other_axis(axis):
    return 'row' if axis=='col' else 'col'

def is_superset(a, b):
    return set(a).issuperset(b)

def overlap(a, b):
    return not set(a).isdisjoint(b)

############################
# Visual feature helpers
############################
_Bbox = namedtuple('bbox', ['page', 'top','bottom','left','right'], verbose = False)
def _bbox_from_span(span):
    if isinstance(span, TemporarySpan) and span.has_visual_features():
        return _Bbox(
                    span.get_attrib_tokens('page')[0],
                    min(span.get_attrib_tokens('top')), 
                    max(span.get_attrib_tokens('bottom')),
                    min(span.get_attrib_tokens('left')),
                    max(span.get_attrib_tokens('right')))
    else:
        return None
    
def _bbox_from_phrase(phrase):
    # TODO: this may have issues where a phrase is linked to words on different pages
    if isinstance(phrase, Phrase) and phrase.is_visual():
        return _Bbox(
                    phrase.page[0],
                    min(phrase.top), 
                    max(phrase.bottom),
                    min(phrase.left),
                    max(phrase.right))
    else:
        return None

def _bbox_horz_aligned(box1, box2):
    """
    Returns true if the vertical center point of either span is within the 
    vertical range of the other
    """
    if not (box1 and box2): return False
    center1 = (box1.bottom + box1.top)/2.0
    center2 = (box2.bottom + box2.top)/2.0
    return ((center1 >= box2.top and center1 <= box2.bottom) or
            (center2 >= box1.top and center2 <= box1.bottom))

def _bbox_vert_aligned(box1, box2):
    """
    Returns true if the horizontal center point of either span is within the 
    horizontal range of the other
    """
    if not (box1 and box2): return False
    center1 = (box1.right + box1.left)/2.0
    center2 = (box2.right + box2.left)/2.0
    return ((center1 >= box2.left and center1 <= box2.right) or
            (center2 >= box1.left and center2 <= box1.right))

def _bbox_vert_aligned_left(box1, box2):
    """
    Returns true if the left boundary of both boxes is within 2 pts
    """
    if not (box1 and box2): return False
    return abs(box1.left - box2.left) <= 2

def _bbox_vert_aligned_right(box1, box2):
    """
    Returns true if the right boundary of both boxes is within 2 pts
    """
    if not (box1 and box2): return False
    return abs(box1.right - box2.right) <= 2

def _bbox_vert_aligned_center(box1, box2):
    """
    Returns true if the right boundary of both boxes is within 5 pts
    """
    if not (box1 and box2): return False
    return abs((box1.right + box1.left)/2.0 - (box2.right + box2.left)/2.0) <= 5

def is_horz_aligned(c):
    return (all([_bbox_from_span(c[i]).page is not None and 
                 _bbox_horz_aligned(_bbox_from_span(c[i]), _bbox_from_span(c[0]))
                 for i in range(len(c))]))

def is_vert_aligned(c):
    return (all([_bbox_from_span(c[i]).page is not None and 
                _bbox_vert_aligned(_bbox_from_span(c[i]), _bbox_from_span(c[0]))
                for i in range(len(c))]))

def is_vert_aligned_left(c):
    return (all([_bbox_from_span(c[i]).page is not None and 
            _bbox_vert_aligned_left(_bbox_from_span(c[i]), _bbox_from_span(c[0]))
            for i in range(len(c))]))

def is_vert_aligned_right(c):
    return (all([_bbox_from_span(c[i]).page is not None and 
            _bbox_vert_aligned_right(_bbox_from_span(c[i]), _bbox_from_span(c[0]))
            for i in range(len(c))]))

def is_vert_aligned_center(c):
    return (all([_bbox_from_span(c[i]).page is not None and 
            _bbox_vert_aligned_center(_bbox_from_span(c[i]), _bbox_from_span(c[0]))
            for i in range(len(c))]))

def same_page(c):
    return (all([_bbox_from_span(c[i]).page is not None and 
                 _bbox_from_span(c[i]).page == _bbox_from_span(c[0]).page
                 for i in range(len(c))]))

def get_horz_aligned_ngrams(c, attrib='words', n_min=1, n_max=1, lower=True):
    # TODO: this currently looks only in current table; 
    #   precompute over the whole document/page instead
    # TODO: this currently aligns based on phrases, not words
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    for span in spans:
        if span.parent.table is None: continue
        for phrase in span.parent.table.phrases:
            if (_bbox_horz_aligned(_bbox_from_phrase(phrase), _bbox_from_span(span)) and
                 phrase is not span.parent):
                for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                    yield ngram

def get_vert_aligned_ngrams(c):
    # TODO
    return

def get_vert_aligned_left_ngrams(c):
    # TODO
    return

def get_vert_aligned_right_ngrams(c):
    # TODO
    return

def get_vert_aligned_center_ngrams(c):
    # TODO
    return

def get_visual_header_ngrams(c, axis=None):
    # TODO
    return

def get_visual_distance(c, axis=None):
    # TODO
    return

def _assign_alignment_features(phrases_by_key, align_type):
    for key, phrases in phrases_by_key.iteritems():
        if len(phrases) == 1: continue
        context_lemmas = set()
#         print 'For group', align_type, '\t'.join(p.text for p in phrases)
        for p in phrases:
#             print 'Adding', context_lemmas, 'to', p
            p._aligned_lemmas.update(context_lemmas)
            # update lemma context for upcoming phrases in the group
            if len(p.lemmas) < 7:
                new_lemmas = [lemma.lower() for lemma in p.lemmas if lemma.isalpha()]
#                 if new_lemmas: print '++Lemmas for\t', p, context_lemmas
                context_lemmas.update(new_lemmas)
                context_lemmas.update(align_type + lemma for lemma in new_lemmas)
#         print context_lemmas
    
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
            phrase.bbox = _bbox_from_phrase(phrase)
            phrase.yc = (phrase.bbox.top + phrase.bbox.bottom)/2
            phrase.x0 = phrase.bbox.left
            phrase.x1 = phrase.bbox.right
            phrase.xc = (phrase.x0 + phrase.x1) / 2
            # index current phrase by different alignment keys
            yc_aligned[phrase.yc].append(phrase)
            x0_aligned[phrase.x0].append(phrase)
            x1_aligned[phrase.x1].append(phrase)
            xc_aligned[phrase.xc].append(phrase)
        for l in yc_aligned.itervalues(): l.sort(key=lambda p:p.yc)
        for l in x0_aligned.itervalues(): l.sort(key=lambda p:p.x0)
        for l in x1_aligned.itervalues(): l.sort(key=lambda p:p.x1)
        for l in xc_aligned.itervalues(): l.sort(key=lambda p:p.xc)
        _assign_alignment_features(yc_aligned, 'Y_')
        _assign_alignment_features(x0_aligned, 'LEFT_')
        _assign_alignment_features(x1_aligned, 'RIGHT_')
        _assign_alignment_features(xc_aligned, 'CENTER_')
    
    
def get_visual_aligned_lemmas(span):
    phrase = span.parent
    doc = phrase.document
    # cache features for the entire document
    _preprocess_visual_features(doc)
    
    for aligned_lemma in phrase._aligned_lemmas:
        yield aligned_lemma
        
def get_aligned_lemmas(span):
    return set(get_visual_aligned_lemmas(span))

