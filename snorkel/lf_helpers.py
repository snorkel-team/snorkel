from .models import SnorkelSession, TemporarySpan, Span, Cell, Phrase
from itertools import chain
from utils import tokens_to_ngrams
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


def get_between_ngrams(c, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    """
    Get the ngrams _between_ two unary Spans of a binary-Span Candidate, where
    both share the same parent Context.
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param case_sensitive: If false, all ngrams will be returned in lower case
    """
    if len(c.get_arguments()) != 2:
        raise ValueError("Only applicable to binary Candidates")
    span0 = c[0]
    span1 = c[1]
    if span0.parent != span1.parent:
        raise ValueError("Only applicable to Candidates where both spans are from the same immediate Context.")
    distance = abs(span0.get_word_start() - span1.get_word_start())
    if span0.get_word_start() < span1.get_word_start():
        for ngram in get_right_ngrams(span0, window=distance-1, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive):
            yield ngram
    else: # span0.get_word_start() > span1.get_word_start()
        for ngram in get_left_ngrams(span1, window=distance-1, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive):
            yield ngram

# TODO: define for compatibility with master (can call on *_*_ngrams)
# def get_left_tokens()
# def get_right_tokens()

def get_left_ngrams(c, window=3, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    """
    Get the ngrams within a window to the _left_ of the Candidate from its parent Context.
    For higher-arity Candidates, defaults to the _first_ argument.
    :param window: The number of tokens to the left of the first argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param case_sensitive: If false, all ngrams will be returned in lower case
    """
    span = c if isinstance(c, TemporarySpan) else c[0] 
    i    = span.get_word_start()
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    for ngram in tokens_to_ngrams(map(f, span.parent._asdict()[attrib][max(0, i-window):i]), n_min=n_min, n_max=n_max):
        yield ngram


def get_right_ngrams(c, window=3, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    """
    Get the ngrams within a window to the _right_ of the Candidate from its parent Context.
    For higher-arity Candidates, defaults to the _last_ argument.
    :param window: The number of tokens to the left of the first argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param case_sensitive: If false, all ngrams will be returned in lower case
    """
    span = c if isinstance(c, TemporarySpan) else c[-1]
    i    = span.get_word_end()
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    for ngram in tokens_to_ngrams(map(f, span.parent._asdict()[attrib][i+1:i+1+window]), n_min=n_min, n_max=n_max):
        yield ngram


def contains_token(c, tok, attrib='words', case_sensitive=False):
    """
    Return True if any of the contituent Spans contain the given token
    :param tok: The token being searched for
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param case_sensitive: If false, all ngrams will be returned in lower case
    """
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return f(tok) in set(chain.from_iterable(map(f, span.get_attrib_tokens(attrib))
        for span in spans))


def contains_regex(c, rgx=None, attrib='words', sep=" ", case_sensitive=False):
    """
    Return True if any of the contituent Spans contain the given regular expression
    :param rgx: The regex being searched for
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param sep: The separator to be used in concatening the retrieved tokens
    :param case_sensitive: If false, all ngrams will be returned in lower case
    """
    spans = [c] if isinstance(c, TemporarySpan) else c.get_arguments()
    r = re.compile(rgx, flags=re.I if not case_sensitive else 0)
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


def get_phrase_ngrams(span, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    """
    Get the ngrams that are in the Phrase of the given span, not including itself.
    :param span: The span whose Phrase is being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param case_sensitive: If false, all ngrams will be returned in lower case
    """
    if not isinstance(span, TemporarySpan):
        raise ValueError("Handles Span-type Candidate arguments only")
    for ngram in get_left_ngrams(span, window=100, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive):
        yield ngram
    for ngram in get_right_ngrams(span, window=100, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive):
        yield ngram


def get_cell_ngrams(span, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    """
    Get the ngrams that are in the Cell of the given span, not including itself.
    :param span: The span whose Cell is being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param case_sensitive: If false, all ngrams will be returned in lower case
    """
    if not isinstance(span, TemporarySpan):
        raise ValueError("Handles Span-type Candidate arguments only")
    if (not isinstance(span.parent, Phrase) or
        span.parent.cell is None): return
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    for ngram in (chain.from_iterable(tokens_to_ngrams(map(f, getattr(phrase,attrib)), n_min=n_min, n_max=n_max) 
        for phrase in span.parent.cell.phrases)):
        yield ngram


def get_neighbor_cell_ngrams(span, dist=1, directions=False, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    """
    Get the ngrams from all Cells that are within a given Cell distance in one direction from the given span  
    :param span: The span whose neighbor Cells are being searched
    :param dist: The Cell distance within which a neighbor Cell must be to be considered
    :param directions: A Boolean expressing whether or not to return the direction of each ngram
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param case_sensitive: If false, all ngrams will be returned in lower case
    """
    # TODO: Fix this to be more efficient (optimize with SQL query)
    if not isinstance(span, TemporarySpan):
        raise ValueError("Handles Span-type Candidate arguments only")
    if (not isinstance(span.parent, Phrase) or
        span.parent.cell is None): return
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    for phrase in span.parent.table.phrases:
        if phrase.cell is None:
            continue
        row_diff = abs(phrase.row_num - span.parent.row_num)
        col_diff = abs(phrase.col_num - span.parent.col_num)
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
                for ngram in tokens_to_ngrams(map(f, getattr(phrase, attrib)), n_min=n_min, n_max=n_max):
                    yield (ngram, direction)
            else: 
                for ngram in tokens_to_ngrams(map(f, getattr(phrase, attrib)), n_min=n_min, n_max=n_max):
                    yield  ngram


def get_row_ngrams(span, infer=False, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    """
    Get the ngrams from all Cells that are in the same row as the given span 
    :param span: The span whose neighbor Cells are being searched
    :param infer: If True, then if a Cell is empty, use the contents from the first non-empty Cell above it
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param case_sensitive: If false, all ngrams will be returned in lower case
    """
    for ngram in _get_axis_ngrams(span, axis='row', infer=infer, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive):
        yield ngram


def get_col_ngrams(span, infer=False, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    """
    Get the ngrams from all Cells that are in the same column as the given span 
    :param span: The span whose neighbor Cells are being searched
    :param infer: If True, then if a Cell is empty, use the contents from the first non-empty Cell to the left of it
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param case_sensitive: If false, all ngrams will be returned in lower case
    """
    for ngram in _get_axis_ngrams(span, axis='col', infer=infer, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive):
        yield ngram


def get_aligned_ngrams(span, infer=False, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    """
    Get the ngrams from all Cells that are in the same row or column as the given span 
    :param span: The span whose neighbor Cells are being searched
    :param infer: Refer to get_[row/col]_ngrams for description
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param case_sensitive: If false, all ngrams will be returned in lower case
    """
    for ngram in get_row_ngrams(span, infer=infer, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive):
        yield ngram
    for ngram in get_col_ngrams(span, infer=infer, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive):
        yield ngram

# TODO: write this LF helper (get furthest north and west cell's ngrams)
# def get_head_ngrams
# ...sorted(_get_aligned_cells(cell, axis, infer=False), key=lambda x: getattr(x,axis_name))[0]

def _get_axis_ngrams(span, axis, infer=False, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    if not isinstance(span, TemporarySpan):
        raise ValueError("Handles Span-type Candidate arguments only")
    if (not isinstance(span.parent, Phrase) or
        span.parent.cell is None): return
    # TODO: optimize this with SQL query 
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    for cell in _get_aligned_cells(span.parent.cell, axis, infer=infer):
        for phrase in cell.phrases:
            for ngram in tokens_to_ngrams(map(f, getattr(phrase, attrib)), n_min=n_min, n_max=n_max):
                yield ngram 

# WITHOUT SQL:
def _get_aligned_cells(root_cell, axis, infer=False):
    axis_name = axis + '_num'
    other_axis = 'row' if axis=='col' else 'col'
    aligned_cells = [cell for cell in root_cell.table.cells
        if getattr(cell, axis_name) == getattr(root_cell, axis_name)
        and cell != root_cell]
    return [_get_nonempty_cell(cell, other_axis) for cell in aligned_cells] if infer else aligned_cells 

# WITH SQL:
# TODO: use getattr for row_num/col_num 
# def _get_aligned_cells(root_cell, axis, infer=False):
#     session = SnorkelSession.object_session(root_cell)
#     if axis == 'row':
#         aligned_cells = session.query(Cell).filter(
#             Cell.table == root_cell.table).filter(
#             Cell.row_num == root_cell.row_num).filter(
#             Cell.id != root_cell.id).all()
#         other_axis = 'col'
#     else:
#         aligned_cells = session.query(Cell).filter(
#             Cell.table == root_cell.table).filter(
#             Cell.col_num == root_cell.col_num).filter(
#             Cell.id != root_cell.id).all()
#         other_axis = 'row'
#     return [_get_nonempty_cell(cell, other_axis) for cell in aligned_cells] if infer else aligned_cells 

# WITHOUT SQL:
def _get_nonempty_cell(root_cell, axis):
    axis_name = axis + '_num'
    other_axis = 'row' if axis=='col' else 'col'
    other_axis_name = other_axis + '_num'
    if root_cell.text or getattr(root_cell, other_axis_name) == 0:
        return root_cell
    else:
        neighbor_cell = [cell for cell in root_cell.table.cells
            if getattr(cell, axis_name) == getattr(root_cell, axis_name)
            and getattr(cell, other_axis_name) == getattr(root_cell, other_axis_name) - 1]
        return _get_nonempty_cell(neighbor_cell[0], axis)
        
# WITH SQL:
# def _get_nonempty_cell(root_cell, axis):
#     axis_name = axis + '_num'
#     other_axis = 'row' if axis=='col' else 'col'
#     other_axis_name = other_axis + '_num'
#     if root_cell.text or getattr(root_cell, other_axis_name) == 0:
#         return root_cell
#     else:
#         session = SnorkelSession.object_session(root_cell)
#         return session.query(Cell).filter(
#             Cell.table_id == root_cell.table_id).filter(
#             Cell.row_num == root_cell.row_num).filter(
#             Cell.col_num < root_cell.col_num).order_by(
#             -Cell.col_num).first()
