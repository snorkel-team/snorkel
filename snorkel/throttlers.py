class Throttler(object):
    def apply(self, span1, span2):
        """
        Return True if the tuple should be allowed. Return False if it should
        be throttled.
        """
        return True

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