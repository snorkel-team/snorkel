def _min_range_diff(a_start, a_end, b_start, b_end, absolute=True):
    # if absolute=True, return the absolute value of minimum magnitude difference
    # if absolute=False, return the raw value of minimum magnitude difference
    return max(0, max(a_end - b_start, b_end - a_start))

def min_row_diff(a, b, absolute=True):
    return _min_range_diff(a.row_start, a.row_end, b.row_start, b.row_end, absolute=absolute)

def min_col_diff(a, b, absolute=True):
    return _min_range_diff(a.col_start, a.col_end, b.col_start, b.col_end, absolute=absolute)

def min_axis_diff(a, b, axis=None, absolute=True):
    if axis == 'row':
        return min_row_diff(a, b, absolute)
    elif axis == 'col':
        return min_col_diff(a, b, absolute)
    else: 
        return min(min_row_diff(a, b, absolute), min_col_diff(a, b, absolute))

def is_row_aligned(a, b):
    return min_row_diff(a, b) == 0

def is_col_aligned(a, b):
    return min_col_diff(a, b) == 0

def is_axis_aligned(a, b, axis=None):
    if axis == 'row':
        return is_row_aligned(a, b)
    elif axis == 'col':
        return is_col_aligned(a, b)
    else:
        return is_row_aligned(a, b) or is_col_aligned(a, b)

def num_rows(a):
    return a.row_start - a.row_end + 1

def num_cols(a):
    return a.col_start - a.col_end + 1