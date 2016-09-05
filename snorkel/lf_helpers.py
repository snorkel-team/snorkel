from .models import Span


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
    chunks = get_text_splits(c)
    if len(chunks) == 5:
        return chunks[2]
    else:
        raise ValueError("Only applicable to binary Candidates")
