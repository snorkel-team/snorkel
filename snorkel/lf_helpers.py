from .models import SnorkelSession, Span, Cell, Phrase
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
        if not isinstance(span, Span):
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
    TODO: write doc_string
    """
    if len(c.get_arguments()) != 2:
        raise ValueError("Only applicable to binary Candidates")
    span0 = c[0]
    span1 = c[1]
    distance = abs(span0.get_word_start() - span1.get_word_start())
    if span0.get_word_start() < span1.get_word_start():
        return get_right_ngrams(span0, window=distance-1, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive)
    else: # span0.get_word_start() > span1.get_word_start()
        return get_left_ngrams(span1, window=distance-1, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive)


def get_left_ngrams(c, window=3, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    """
    Return the ngrams within a window to the _left_ of the Candidate.
    For higher-arity Candidates, defaults to the _first_ argument.
    :param window: The number of tokens to the left of the first argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    span = c if isinstance(c, Span) else c[0] 
    i    = span.get_word_start()
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return [ngram for ngram in tokens_to_ngrams(map(f, span.parent._asdict()[attrib][max(0, i-window):i]), n_min=n_min, n_max=n_max)]


def get_right_ngrams(c, window=3, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    """
    Return the ngrams within a window to the _right_ of the Candidate.
    For higher-arity Candidates, defaults to the _last_ argument.
    :param window: The number of tokens to the right of the last argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    span = c if isinstance(c, Span) else c[-1]
    i    = span.get_word_end()
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return [ngram for ngram in tokens_to_ngrams(map(f, span.parent._asdict()[attrib][i+1:i+1+window]), n_min=n_min, n_max=n_max)]


def contains_token(c, tok, attrib='words', case_sensitive=False):
    """
    Checks if any of the contituent Spans contain a token
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    spans = [c] if isinstance(c, Span) else c.get_arguments()
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return f(tok) in set(chain.from_iterable(map(f, span.get_attrib_tokens(attrib))
        for span in spans))


def contains_regex(c, rgx=None, attrib='words', sep=" ", case_sensitive=False):
    """
    TODO: write documentation here
    """   
    spans = [c] if isinstance(c, Span) else c.get_arguments()
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
def get_phrase_ngrams(span, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    return (get_left_ngrams(span, window=100, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive)
          + get_right_ngrams(span, window=100, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive))


def get_cell_ngrams(span, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    """
    Checks if any of the contituent Spans contain a token
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    if (not isinstance(span, Span) or 
        not isinstance(span.parent, Phrase) or
        span.parent.cell is None): return []
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return list(chain.from_iterable(tokens_to_ngrams(map(f, getattr(phrase,attrib)), n_min=n_min, n_max=n_max) 
        for phrase in span.parent.cell.phrases))


def get_neighbor_cell_ngrams(c, dist=1, directions=False, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    # TODO: Fix this to be more efficient (optimize with SQL query)
    span = c
    if (not isinstance(span, Span) or 
        not isinstance(span.parent, Phrase) or
        span.parent.cell is None): return []
    if directions:
        f = (lambda w: w) if case_sensitive else (lambda w: (w[0].lower(), w[1]))
    else:
        f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    ngrams = []
    for phrase in span.parent.table.phrases:
        if phrase.cell is None:
            continue
        row_diff = abs(phrase.row_num - span.parent.row_num)
        col_diff = abs(phrase.col_num - span.parent.col_num)
        if (row_diff or col_diff) and row_diff + col_diff <= dist:
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
                ngrams.extend([(ngram, direction) for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max)])
            else: 
                ngrams.extend([ngram for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max)]) 
    return map(f, ngrams)


def get_row_ngrams(c, infer=False, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    return _get_axis_ngrams(c, axis='row', infer=infer, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive)


def get_col_ngrams(c, infer=False, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    return _get_axis_ngrams(c, axis='col', infer=infer, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive)


def get_aligned_ngrams(c, infer=False, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    return (get_row_ngrams(c, infer=infer, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive)
          + get_col_ngrams(c, infer=infer, attrib=attrib, n_min=n_min, n_max=n_max, case_sensitive=case_sensitive))

# TODO: write this LF helper (get furthest north and west cell's ngrams)
# def get_head_ngrams
# ...sorted(_get_aligned_cells(cell, axis, infer=False), key=lambda x: getattr(x,axis_name))[0]

def same_document(c):
    return (c[0].parent.document is not None and
            c[1].parent.document is not None and 
            c[0].parent.document == c[1].parent.document)

def same_table(c):
    return (c[0].parent.table is not None and
            c[1].parent.table is not None and 
            c[0].parent.table == c[1].parent.table)

def same_cell(c):
    return (c[0].parent.cell is not None and
            c[1].parent.cell is not None and 
            c[0].parent.cell == c[1].parent.cell)

def same_phrase(c):
    return (c[0].parent is not None and
            c[1].parent is not None and 
            c[0].parent == c[1].parent)


def _get_axis_ngrams(c, axis, infer=False, attrib='words', n_min=1, n_max=1, case_sensitive=False):
    # TODO: optimize this with SQL query
    span = c
    if (not isinstance(span, Span) or 
        not isinstance(span.parent, Phrase) or
        span.parent.cell is None): return []  
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    ngrams = []
    for cell in _get_aligned_cells(span.parent.cell, axis, infer=infer):
        for phrase in cell.phrases:
            ngrams.extend([ngram for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_min=n_min, n_max=n_max)]) 
    return map(f, ngrams)

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
