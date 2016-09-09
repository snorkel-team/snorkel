from .models import Span, Phrase
from itertools import chain
from utils import tokens_to_ngrams

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


def get_between_ngrams(c, attrib='words', n_max=1, case_sensitive=False):
    """
    TODO: write doc_string
    """
    if len(c.get_arguments()) != 2:
        raise ValueError("Only applicable to binary Candidates")
    span0 = c[0]
    span1 = c[1]
    distance = abs(span0.get_word_start() - span1.get_word_start())
    if span0.get_word_start() < span1.get_word_start():
        return get_right_ngrams(span0, window=distance-1, attrib=attrib, n_max=n_max, case_sensitive=case_sensitive)
    else: # span0.get_word_start() > span1.get_word_start()
        return get_left_ngrams(span1, window=distance-1, attrib=attrib, n_max=n_max, case_sensitive=case_sensitive)


def get_left_ngrams(c, window=3, attrib='words', n_max=1, case_sensitive=False):
    """
    Return the ngrams within a window to the _left_ of the Candidate.
    For higher-arity Candidates, defaults to the _first_ argument.
    :param window: The number of tokens to the left of the first argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    span = c if isinstance(c, Span) else c[0] 
    i    = span.get_word_start()
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return [ngram for ngram in tokens_to_ngrams(map(f, span.parent._asdict()[attrib][max(0, i-window):i]), n_max=n_max)]


def get_right_ngrams(c, window=3, attrib='words', n_max=1, case_sensitive=False):
    """
    Return the ngrams within a window to the _right_ of the Candidate.
    For higher-arity Candidates, defaults to the _last_ argument.
    :param window: The number of tokens to the right of the last argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    span = c if isinstance(c, Span) else c[-1]
    i    = span.get_word_end()
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return [ngram for ngram in tokens_to_ngrams(map(f, span.parent._asdict()[attrib][i+1:i+1+window]), n_max=n_max)]


def contains_token(c, tok, attrib='words', case_sensitive=False):
    """
    Checks if any of the contituent Spans contain a token
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    spans = [c] if isinstance(c, Span) else c.get_arguments()
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return f(tok) in set(chain.from_iterable(map(f, span.get_attrib_tokens(attrib))
        for span in spans))


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
def get_phrase_ngrams(span, attrib='words', n_max=1, case_sensitive=False):
    return (get_left_ngrams(span, window=100, attrib=attrib, n_max=n_max, case_sensitive=case_sensitive)
          + get_right_ngrams(span, window=100, attrib=attrib, n_max=n_max, case_sensitive=case_sensitive))


def get_cell_ngrams(span, attrib='words', n_max=1, case_sensitive=False):
    """
    Checks if any of the contituent Spans contain a token
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    if (not isinstance(span, Span) or 
        not isinstance(span.parent, Phrase) or
        span.parent.cell is None): return []
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return list(chain.from_iterable(tokens_to_ngrams(map(f, getattr(phrase,attrib)), n_max=n_max) 
        for phrase in span.parent.cell.phrases))


def get_neighbor_cell_ngrams(c, attrib='words', n_max=1, case_sensitive=False):
    # TODO: Fix this to be more efficient (optimize with SQL query)
    span = c
    if (not isinstance(span, Span) or 
        not isinstance(span.parent, Phrase) or
        span.parent.cell is None): return []
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    ngrams = []
    for phrase in span.parent.table.phrases:
        if phrase.cell is None:
            continue
        row_diff = abs(phrase.row_num - span.parent.row_num)
        col_diff = abs(phrase.col_num - span.parent.col_num)
        # side = ''
        # if col_diff==0:
        #     if 0 < row_diff and row_diff <= dist:
        #         side = "UP"
        #     elif  0 > row_diff and row_diff >= -dist:
        #         side = "DOWN"
        # elif row_diff==0:
        #     if 0 < col_diff and col_diff <= dist:
        #         side = "RIGHT"
        #     elif  0 > col_diff and col_diff >= -dist:
        #         side = "LEFT"
        # if side:
        if row_diff + col_diff == 1:
            ngrams.extend([ngram for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_max=n_max)]) 
    return map(f, ngrams)


def get_row_ngrams(c, attrib='words', n_max=1, case_sensitive=False):
    return _get_axis_ngrams(c, axis='row', attrib=attrib, n_max=n_max, case_sensitive=case_sensitive)


def get_col_ngrams(c, attrib='words', n_max=1, case_sensitive=False):
    return _get_axis_ngrams(c, axis='col', attrib=attrib, n_max=n_max, case_sensitive=case_sensitive)


def get_aligned_ngrams(c, attrib='words', n_max=1, case_sensitive=False):
    return (get_row_ngrams(c, attrib=attrib, n_max=n_max, case_sensitive=case_sensitive)
          + get_col_ngrams(c, attrib=attrib, n_max=n_max, case_sensitive=case_sensitive))


# def head_ngrams(axis, attrib='words', n_max=1, case_sensitive=False, induced=False):
#     head_cell = self.head_cell(axis, induced=induced)
#     ngrams = chain.from_iterable(
#         self._get_phrase_ngrams(phrase, attrib=attrib, n_max=n_max) 
#         for cell in cells for phrase in cell.phrases)
#     return [ngram.lower() for ngram in ngrams] if not case_sensitive else ngrams


# def head_cell(axis, induced=False):
#     if axis not in ('row', 'col'): 
#         raise Exception("Axis must equal 'row' or 'col' ")

#     cells = self._get_aligned_cells(axis=axis)
#     axis_name = axis + '_num'
#     head_cell = sorted(cells, key=lambda x: getattr(x,axis_name))[0]
    
#     if induced and not head_cell.text.isspace():
#         other_axis = 'col' if axis == 'row' else 'row'
#         other_axis_name = other_axis + '_num'
#         # get aligned cells to head_cell that appear before head_cell and aren't empty
#         aligned_cells = [cell for cell in head_cell.table.cells
#                             if getattr(cell,other_axis_name) == getattr(head_cell,other_axis_name)
#                             if getattr(cell,axis_name) < getattr(head_cell,axis_name)
#                             and not cell.text.isspace()]
#         # pick the last cell among the ones identified above
#         aligned_cells = sorted(aligned_cells, key=lambda x: getattr(x,axis_name), reverse=True)
#         if aligned_cells:
#             head_cell = aligned_cells[0]

#     return head_cell


# def neighborhood_ngrams(attrib='words', n_max=3, dist=1, case_sensitive=False):
#     # TODO: Fix this to be more efficient (optimize with SQL query)
#     if self.context.cell is None: return
#     f = lambda x: 0 < x and x <= dist
#     phrases = [phrase for phrase in self.context.table.phrases if
#         phrase.row_num is not None and phrase.col_num is not None and
#         f(abs(phrase.row_num - self.context.row_num) + abs(phrase.col_num - self.context.col_num))]
#     for phrase in phrases:
#         for ngram in slice_into_ngrams(getattr(phrase,attrib), n_max=n_max):
#             yield ngram if case_sensitive else ngram.lower()


def _get_row_cells(span):
    return [cell for cell in _get_aligned_cells(span, axis='row')]


def _get_col_cells(span):
    return [cell for cell in _get_aligned_cells(span, axis='col')]


def _get_aligned_cells(span, axis):
    axis_name = axis + '_num'
    cells = [cell for cell in span.parent.table.cells
        if getattr(cell, axis_name) == getattr(span.parent, axis_name)
        and cell != span.parent.cell]
    return cells


def _get_axis_ngrams(c, axis, attrib='words', n_max=1, case_sensitive=False):
    span = c
    if (not isinstance(span, Span) or 
        not isinstance(span.parent, Phrase) or
        span.parent.cell is None): return []  
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    ngrams = []
    for cell in _get_aligned_cells(span, axis):
        for phrase in cell.phrases:
            ngrams.extend([ngram for ngram in tokens_to_ngrams(getattr(phrase, attrib), n_max=n_max)]) 
    return map(f, ngrams)

