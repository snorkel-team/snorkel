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
        # TODO: this needs to be made symmetric, or the induced span needs to be specified
        if self.axis in ('row', 'col'):
            aligned_cells = get_aligned_cells(span0.parent.cell, self.axis, infer=True)
        if self.axis is None:
            aligned_cells = get_aligned_cells(span0.parent.cell, 'row', infer=True) \
                          + get_aligned_cells(span0.parent.cell, 'col', infer=True)
        return span1.parent.cell in aligned_cells

class SeparatingSpanThrottler(Throttler):
    """Filter spans that are separated by spanning cells

    We look for all the cells in between and check if any of them span the table.
    """
    def __init__(self, align_axis):
        if align_axis not in ('row', 'col'): raise ValueError('Invalid axis: %s' % align_axis)
        self.align_axis = align_axis

    def __call__(self, argtuple):
        # get the cells associated with each span
        span0, span1 = argtuple
        cell0, cell1 = span0.parent.cell, span1.parent.cell

        # figure out which one comes first and which ones comes second
        ax = 'row' if self.align_axis == 'col' else 'row'
        top_cell = cell0 if getattr(cell0, ax).position < getattr(cell1, ax).position else cell1
        bot_cell = cell0 if getattr(cell0, ax).position > getattr(cell1, ax).position else cell1
        
        # get the set of middle cells
        min_ax = getattr(top_cell, ax).position
        max_ax = getattr(bot_cell, ax).position
        middle_cells = [ c for c in span0.parent.table.cells 
                         if min_ax < getattr(c, ax).position < max_ax ]
        if not middle_cells: return False

        # if the two spans are separated by a spanning cell, we skip this pair
        sp_ax = 'row' if ax == 'col' else 'row'
        def _spans(cell, table, axis):
            if axis == 'row' and len([c for c in table.cells if c.row == cell.row]) > 1:
                return False
            if axis == 'col' and len([c for c in table.cells if c.col == cell.col]) > 1:
                return False
            return True

        # print '!!!', span0, span0.parent.cell.row.position, span0.parent.cell.col.position, span1, span1.parent.cell.row.position, span1.parent.cell.col.position
        # for c in middle_cells:
        #     print c.text, c.row.position, c.col.position, len([d for d in span0.parent.table.cells if c.row == d.row]), _spans(c, span0.parent.table, sp_ax)
                             
        return False if any(_spans(c, span0.parent.table, sp_ax) for c in middle_cells) else True

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