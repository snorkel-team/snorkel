from .models import Span
from itertools import chain


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

def get_between_tokens(c, attrib='words'):
    """
    TODO: write doc_string
    """
    if len(c.get_arguments()) != 2:
        raise ValueError("Only applicable to binary Candidates")
    span0 = c[0]
    span1 = c[1]
    distance = abs(span0.get_word_start() - span1.get_word_start())
    if span0.get_word_start() < span1.get_word_start():
        get_right_tokens(c, window=distance-1, attrib=attrib)
    else: # span0.get_word_start() < span1.get_word_start()
        get_right_tokens(c, )


def get_left_tokens(c, window=3, attrib='words'):
    """
    Return the tokens within a window to the _left_ of the Candidate.
    For higher-arity Candidates, defaults to the _first_ argument.
    :param window: The number of tokens to the left of the first argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    span = c[0]
    i    = span.get_word_start()
    return span.parent._asdict()[attrib][max(0, i-window):i]


def get_right_tokens(c, window=3, attrib='words'):
    """
    Return the tokens within a window to the _right_ of the Candidate.
    For higher-arity Candidates, defaults to the _last_ argument.
    :param window: The number of tokens to the right of the last argument to return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    span = c[-1]
    i    = span.get_word_end()
    return span.parent._asdict()[attrib][i+1:i+1+window]


def contains_token(c, tok, attrib='words'):
    """
    Checks if any of the contituent Spans contain a token
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    return tok in set(chain.from_iterable(span.get_attrib_tokens(attrib) for span in c.get_arguments()))


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
