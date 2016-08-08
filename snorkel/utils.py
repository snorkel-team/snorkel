from itertools import chain

def get_as_dict(x):
    """Return an object as a dictionary of its attributes"""
    if isinstance(x, dict):
        return x
    else:
        try:
            return x._asdict()
        except AttributeError:
            return x.__dict__

def sort_X_on_Y(X, Y):
    return [x for (y,x) in sorted(zip(Y,X), key=lambda t : t[0])]

def corenlp_cleaner(words):
    d = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{',
       '-RSB-': ']', '-LSB-': '['}
    return map(lambda w: d[w] if w in d else w, words)

def split_html_attrs(attrs):
    """
    Given an iterable object of (attr, values) pairs, returns a list of separated
    "attr=value" strings
    """
    html_attrs = []
    for a in attrs:
        attr = a[0]
        values = [v.split(';') for v in a[1]] if isinstance(a[1],list) else [a[1].split(';')]
        for i in range(len(values)):
            while isinstance(values[i], list):
                values[i] = values[i][0]
        html_attrs += ["=".join([attr,val]) for val in values]
        # html_attrs += ["=".join([attr,val]) for val in chain.from_iterable(values)]
        # if not isinstance(values, list):

    return html_attrs

def slice_into_ngrams(tokens, n_max=3, delim='_'):
    N = len(tokens)
    for root in range(N):
        for n in range(min(n_max, N - root)):
            yield delim.join(tokens[root:root+n+1])
