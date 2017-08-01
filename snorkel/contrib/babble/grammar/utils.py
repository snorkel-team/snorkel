def is_cat(label):
    """
    Returns true iff the given label is a category (non-terminal), i.e., is
    marked with an initial '$'.
    """
    return label.startswith('$')

def is_optional(label):
    """
    Returns true iff the given RHS item is optional, i.e., is marked with an
    initial '?'.
    """
    return label.startswith('?') and len(label) > 1
