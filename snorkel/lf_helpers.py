from .models import SnorkelSession, TemporarySpan, Span, Cell, Phrase
from itertools import chain
from utils import tokens_to_ngrams
from collections import namedtuple
import re


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
    Returns the text of c's parent context with c's unary spans replaced with tags {{A}}, {{B}}, etc.
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
    return [ngram for ngram in get_between_ngrams(c, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower)]

def get_between_ngrams(c, attrib='words', n_min=1, n_max=1, lower=True):
    """
    Get the ngrams _between_ two unary Spans of a binary-Span Candidate, where
    both share the same parent Context.
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If false, all ngrams will be returned in lower case
    """
    if len(c.get_arguments()) != 2:
        raise ValueError("Only applicable to binary Candidates")
    span0 = c[0]
    span1 = c[1]
    if span0.parent != span1.parent:
        raise ValueError("Only applicable to Candidates where both spans are from the same immediate Context.")
    distance = abs(span0.get_word_start() - span1.get_word_start())
    if span0.get_word_start() < span1.get_word_start():
        for ngram in get_right_ngrams(span0, window=distance-1, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram
    else: # span0.get_word_start() > span1.get_word_start()
        for ngram in get_left_ngrams(span1, window=distance-1, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram


# TODO: replace in tutorials with get_left_ngrams and delete this
def get_left_tokens(c, window=3, attrib='words', n_min=1, n_max=1, lower=True):
    """ An alias for get_left_ngrams maintained for backwards compatibility. """
    return [ngram for ngram in get_left_ngrams(c, window=window, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower)]


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
    i    = span.get_word_start()
    for ngram in tokens_to_ngrams(span.parent._asdict()[attrib][max(0, i-window):i], n_min=n_min, n_max=n_max, lower=lower):
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
    for ngram in tokens_to_ngrams(span.parent._asdict()[attrib][i+1:i+1+window], n_min=n_min, n_max=n_max, lower=lower):
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
    return (all([c[i].parent.document is not None
        and c[i].parent.document==c[0].parent.document for i in range(len(c.get_arguments()))]))


def same_table(c):
    """
    Return True if all Spans in the given candidate are from the same Table.
    :param c: The candidate whose Spans are being compared
    """
    return (all([c[i].parent.table is not None
        and c[i].parent.table==c[0].parent.table for i in range(len(c.get_arguments()))]))


def same_row(c):
    """
    Return True if all Spans in the given candidate are from the same Row.
    :param c: The candidate whose Spans are being compared
    """
    return (all([c[i].parent.table is not None
        and c[i].parent.row_num==c[0].parent.row_num for i in range(len(c.get_arguments()))]))


def same_col(c):
    """
    Return True if all Spans in the given candidate are from the same Col.
    :param c: The candidate whose Spans are being compared
    """
    return (all([c[i].parent.table is not None
        and c[i].parent.col_num==c[0].parent.col_num for i in range(len(c.get_arguments()))]))


def same_cell(c):
    """
    Return True if all Spans in the given candidate are from the same Cell.
    :param c: The candidate whose Spans are being compared
    """
    return (all([c[i].parent.cell is not None
        and c[i].parent.cell==c[0].parent.cell for i in range(len(c.get_arguments()))]))


def same_phrase(c):
    """
    Return True if all Spans in the given candidate are from the same Phrase.
    :param c: The candidate whose Spans are being compared
    """
    return (all([c[i].parent is not None
        and c[i].parent==c[0].parent for i in range(len(c.get_arguments()))]))


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
            if abs(phrase.phrase_id - span.parent.phrase_id) <= d and phrase != span.parent]):
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
        if span.parent.cell is not None:
            for ngram in chain.from_iterable([tokens_to_ngrams(getattr(phrase,attrib), n_min=n_min, n_max=n_max, lower=lower) \
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
        if span.parent.row_num is not None and span.parent.col_num is not None:
            root_cell = span.parent.cell
            for phrase in chain.from_iterable([root_cell.row.phrases, root_cell.col.phrases]):
                row_diff = abs(phrase.row_num - root_cell.row.position)
                col_diff = abs(phrase.col_num - root_cell.col.position)
                if (row_diff or col_diff) and not (row_diff and col_diff) and row_diff + col_diff <= dist:
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


def get_row_ngrams(c, direct=True, infer=False, attrib='words', n_min=1, n_max=1, lower=True):
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
        for ngram in _get_axis_ngrams(span, axis='row', direct=direct, infer=infer, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram


def get_col_ngrams(c, direct=True, infer=False, attrib='words', n_min=1, n_max=1, lower=True):
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
        for ngram in _get_axis_ngrams(span, axis='col', direct=direct, infer=infer, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram


def get_aligned_ngrams(c, direct=True, infer=False, attrib='words', n_min=1, n_max=1, lower=True):
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
        for ngram in get_row_ngrams(span, direct=direct, infer=infer, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
            yield ngram
        for ngram in get_col_ngrams(span, direct=direct, infer=infer, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
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
    return sorted(aligned_cells, key=lambda x: getattr(x, other_axis + '_num'))[0] if aligned_cells else []


def _get_axis_ngrams(span, axis, direct=True, infer=False, attrib='words', n_min=1, n_max=1, lower=True):
    for ngram in get_phrase_ngrams(span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower):
        yield ngram
    if (span.parent.cell is not None):
        for phrase in _get_aligned_phrases(span.parent, axis, direct=direct, infer=infer):
            for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max, lower=lower):
                yield ngram


def _get_aligned_cells(root_cell, axis, direct=True, infer=False):
    aligned_cells = [cell for cell in root_cell.table.cells
        if getattr(cell, axis) == getattr(root_cell, axis)
        and cell != root_cell]
    return [_infer_cell(cell, _other_axis(axis), direct=direct, infer=infer) \
        for cell in aligned_cells] if infer else aligned_cells


def _get_aligned_phrases(root_phrase, axis, direct=True, infer=False):
    # TODO: There seems to be some discrepency in the order of this list comprehension.
    # This is what is needed for Python 2.7.
    return [phrase for cell in getattr(root_phrase, axis).cells \
                for phrase in _infer_cell(cell, _other_axis(axis), direct, infer).phrases \
                    if phrase!=root_phrase]

PhantomCell = namedtuple('PhantomCell','phrases')
def _infer_cell(root_cell, axis, direct, infer):
    # NOTE: not defined for direct = False and infer = False
    # TODO: Fix this hack; checking for len(text)==9 checks if cell is "<td></td>"
    empty = len(root_cell.text) == 9
    edge = getattr(root_cell, _other_axis(axis)).position == 0
    if direct and (not empty or edge or not infer):
        return root_cell
    else:
        if edge or not empty:
            return PhantomCell(phrases=[]) 
        else:
            neighbor_cells = [cell for cell in root_cell.table.cells
                if getattr(cell, axis) == getattr(root_cell, axis)
                and getattr(cell, _other_axis(axis)).position == getattr(root_cell, _other_axis(axis)).position - 1]
            return _infer_cell(neighbor_cells[0], axis, direct=True, infer=True)

def _other_axis(axis):
    return 'row' if axis=='col' else 'col'

def overlap(A, B):
    return not set(A).isdisjoint(B)