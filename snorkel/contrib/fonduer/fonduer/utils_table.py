from builtins import range
import itertools


def _min_range_diff(a_start, a_end, b_start, b_end, absolute=True):
    # if absolute=True, return the absolute value of minimum magnitude difference
    # if absolute=False, return the raw value of minimum magnitude difference
    # TODO: move back to efficient implementation once it sees that min_range_diff(3,3,2,3) = 0 
    # return max(0, max(a_end - b_start, b_end - a_start))
    f = lambda x: (abs(x) if absolute else x)
    return min([f(ii[0] - ii[1]) for ii in itertools.product(
        list(range(a_start, a_end + 1)), list(range(b_start, b_end + 1)))])


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


def is_row_aligned(a, b, spread=[0, 0]):
    return min_row_diff(a, b) in range(spread[0], spread[1] + 1)


def is_col_aligned(a, b, spread=[0, 0]):
    return min_col_diff(a, b) in range(spread[0], spread[1] + 1)


def is_axis_aligned(a, b, axis=None, spread=[0, 0]):
    if axis == 'row':
        return is_row_aligned(a, b, spread=spread)
    elif axis == 'col':
        return is_col_aligned(a, b, spread=spread)
    else:
        return is_row_aligned(a, b, spread=spread) or is_col_aligned(a, b, spread=spread)


def num_rows(a):
    return a.row_start - a.row_end + 1


def num_cols(a):
    return a.col_start - a.col_end + 1
