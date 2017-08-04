###

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

###

def sems0(sems):
    return sems[0]

def sems1(sems):
    return sems[1]

def sems_in_order(sems):
    return tuple(sems)

def sems_reversed(sems):
    return tuple(reversed(sems))

def flip_dir(dir_):
    if dir_ == '.right':
        return '.left'
    elif dir_ == '.left':
        return '.right'
    else:
        raise Exception