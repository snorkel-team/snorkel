from lf_helpers import get_aligned_cells

class Throttler(object):
    def __call__(self, argtuple):
        """
        Return True if the tuple should be allowed. Return False if it should
        be throttled.
        """
        return NotImplementedError()

class AlignmentThrottler(Throttler):
    """Filters spans that are not aligned"""
    def __init__(self, axis=None, infer=False):
        if axis not in ('row', 'col', None): raise ValueError('Invalid axis: %s' % axis)
        self.axis = axis
        self.infer = infer

    def __call__(self, argtuple):
        if self.infer:
            return self._apply_infer(*argtuple)
        else:
            return self._apply_normal(*argtuple)

    def _apply_normal(self, span0, span1):
        if self.axis == 'row':
            return span0.parent.cell.row.position == span1.parent.cell.row.position
        elif self.axis == 'col':
            return span0.parent.cell.col.position == span1.parent.cell.col.position
        else:
            return span0.parent.cell.row.position == span1.parent.cell.row.position \
                or span0.parent.cell.col.position == span1.parent.cell.col.position

    def _apply_infer(self, span0, span1):
        # TODO: this needs to be made symmetric
        if self.axis in ('row', 'col'):
            aligned_cells = get_aligned_cells(span0.parent.cell, self.axis, infer=True)
        if self.axis is None:
            aligned_cells = get_aligned_cells(span0.parent.cell, 'row', infer=True) \
                          + get_aligned_cells(span0.parent.cell, 'col', infer=True)
        return span1.parent.cell in aligned_cells

# def HeaderSpanThrottler(Throttler):
#     """Filters spans that are not

# Reference derivative class (actual copy stored in hardware_utils.py)
# class PartThrottler(Throttler):
#     """
#     Removes candidates unless the part is not in a table, or the part aligned
#     temperature are not aligned.
#     """
#     def apply(self, part_span, temp_span):
#         """
#         Returns True is the tuple passes, False if it should be throttled
#         """
#         return part_span.parent.table is None or self.aligned(part_span, temp_span)

#     def aligned(self, span1, span2):
#         return (span1.parent.table == span2.parent.table and
#             (span1.parent.row_num == span2.parent.row_num or
#              span1.parent.col_num == span2.parent.col_num))