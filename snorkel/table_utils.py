import itertools

def min_range_diff(a_start, a_end, b_start, b_end, absolute=True):
    # if absolute=True, return the absolute value of minimum magnitude difference
    # if absolute=False, return the raw value of minimum magnitude difference
    f = lambda x: (abs(x) if absolute else x)
    return min([f(ii[0] - ii[1]) for ii in itertools.product(
        range(a_start, a_end + 1),
        range(b_start, b_end + 1))], key=abs)