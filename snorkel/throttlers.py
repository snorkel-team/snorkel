from lf_helpers import get_aligned_cells, cell_spans
from utils_table import is_axis_aligned

class Throttler(object):
    def __call__(self, argtuple):
        """
        Return True if the tuple should be allowed. Return False if it should
        be throttled.
        """
        return NotImplementedError()

class AlignmentThrottler(Throttler):
    """Filters spans that are not aligned"""
    def __init__(self, axis=None, infer=False, infer_pivot=0):
        if axis not in ('row', 'col', None): raise ValueError('Invalid axis: %s' % axis)
        self.axis = axis
        self.infer = infer
        self.infer_pivot = infer_pivot

    def __call__(self, argtuple):
        if self.infer:
            return self._apply_infer(*argtuple)
        else:
            return self._apply_normal(*argtuple)

    def _apply_normal(self, span0, span1):
        return is_axis_aligned(span0.parent.cell, span1.parent.cell, self.axis)

    def _apply_infer(self, span0, span1):
        pivot_span = span0 if self.infer_pivot == 0 else span1
        other_span = span1 if self.infer_pivot == 0 else span0
        if self.axis in ('row', 'col'):
            aligned_cells = get_aligned_cells(pivot_span.parent.cell, self.axis, infer=True)
        if self.axis is None:
            aligned_cells = get_aligned_cells(pivot_span.parent.cell, 'row', infer=True) \
                          + get_aligned_cells(pivot_span.parent.cell, 'col', infer=True)
        return other_span.parent.cell in aligned_cells

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
        if getattr(cell0, ax + '_end') < getattr(cell1, ax + '_start'):
            top_cell, bot_cell = cell0, cell1
        elif getattr(cell1, ax + '_end') < getattr(cell0, ax + '_start'):
            top_cell, bot_cell = cell1, cell0
        else:
            return False
        
        # get the set of middle cells
        min_ax = getattr(top_cell, ax + '_end')
        max_ax = getattr(bot_cell, ax + '_start')
        middle_cells = [ c for c in span0.parent.table.cells 
                         if min_ax < getattr(c, ax + '_start') < max_ax
                         or min_ax < getattr(c, ax + '_end') < max_ax ]
        if not middle_cells: return True # cells are adjacent

        # if the two spans are separated by a spanning cell, we skip this pair
        sp_ax = 'row' if ax == 'col' else 'row'                             
        return False if any(cell_spans(c, span0.parent.table, sp_ax) for c in middle_cells) else True

class OrderingThrottler(Throttler):
    """Filters pairs of spans according to which one comes first"""
    def __init__(self, axis, first=0):
        if axis not in ('row', 'col'): raise ValueError('Invalid axis: %s' % axis)
        if first not in (0,1): raise ValueError('Invalid span index: %d' % first)
        self.axis = axis
        self.first = first

    def __call__(self, argtuple):
        # get the cells associated with each span
        span0, span1 = argtuple
        first_span = span0 if self.first == 0 else span1
        second_span = span1 if self.first == 0 else span0

        cell0, cell1 = first_span.parent.cell, second_span.parent.cell
        ax = 'row' if self.axis == 'col' else 'col'

        return True if getattr(cell0, ax + '_end') <= getattr(cell1, ax + '_start') \
            else False

class OverlapThrottler(Throttler):
    """Only keeps pairs of spans that overlap"""
    def __init__(self):
        pass

    def __call__(self, argtuple):
        # get the cells associated with each span
        span0, span1 = argtuple

        # if hasattr(span0.parent, 'cell') and hasattr(span1.parent, 'cell'):
        #     if span0.parent.cell != span1.parent.cell: return False

        if span0.parent != span1.parent: return False            
        start0, end0 = span0.char_start, span0.char_end
        start1, end1 = span1.char_start, span1.char_end
        return True if start1 <= start0 <= end1 or start1 <= end0 <= end1 else False

class WordLengthThrottler(Throttler):
    """Filter spans by lnumber of words"""
    def __init__(self, op, idx, lim):
        if op not in ('min', 'max'): raise ValueError('Invalid operation: %s' % op)
        if idx not in (0,1): raise ValueError('Invalid span index: %d' % idx)
        self.op = op
        self.idx = idx
        self.lim = lim

    def __call__(self, argtuple):
        # get the cells associated with each span
        span0, span1 = argtuple
        span = span0 if self.idx == 0 else span1

        if self.op == 'max':
            if len(span.get_span().split()) >= self.lim: 
                return False
            else:
                return True
        elif self.op == 'min':
            if len(span.get_span().split()) <= self.lim: 
                return False
            else:
                return True

class CombinedThrottler(Throttler):
    """Filters pairs if they pass all given throttlers"""
    def __init__(self, throttlers):
        self.throttlers = throttlers

    def __call__(self, argtuple):
        return True if all(t(argtuple) for t in self.throttlers) else False

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