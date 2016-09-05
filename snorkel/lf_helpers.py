

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


def get_tagged_text(c):
    """
    Returns the text of c's parent context with c's unary spans replaced with tags {{A}}, {{B}}, etc.
    A convenience method for writing LFs based on e.g. regexes.
    """
    s0, e0 = c.span0.char_start, c.span0.char_end
    s1, e1 = c.span1.char_start, c.span1.char_end
    t = c.span0.context.text
    if s1 >= e0:
        return t[:s0] + r'{{A}}' + t[e0+1:s1] + r'{{B}}' + t[e1+1:]
    else:
        return t[:s1] + r'{{B}}' + t[e1+1:s0] + r'{{A}}' + t[e0+1:]

def get_text_between(c):
    c0_start, c0_end = c.span0.char_start, c.span0.char_end
    c1_start, c1_end = c.span1.char_start, c.span1.char_end
    
    if c0_end <= c1_start:
        return c.span0.context.text[c0_end+1:c1_start], False
    else:
        return c.span0.context.text[c1_end+1:c0_start], True
