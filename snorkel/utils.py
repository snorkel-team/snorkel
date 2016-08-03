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
    html_attrs = []
    for a in attrs:
        attr = a[0]
        values = a[1]
        if isinstance(values, list):
            html_attrs += ["=".join([attr,val]) for val in values]
        else:
            html_attrs += ["=".join([attr,values])]
    return html_attrs
