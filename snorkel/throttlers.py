from snorkel.lf_helpers import *

class Throttler(object):
    def apply(self, span1, span2):
        """
        Return True if the tuple should be allowed. Return False if it should
        be throttled.
        """
        return True

class PartTempThrottler(Throttler):
    """
    Removes candidates unless the part is not in a table, or the part aligned
    temperature are not aligned.
    """
    def apply(self, part_span, temp_span):
        """
        Returns True is the tuple passes, False if it should be throttled
        """
        return part_span.parent.table is None or self.aligned(part_span, temp_span)

    def aligned(self, span1, span2):
        return  (span1.parent.table == span2.parent.table and
            (span1.parent.row_num == span2.parent.row_num or span1.parent.col_num == span2.parent.col_num))

class PartCurrentThrottler(Throttler):
    """
    Removes candidates unless the part is not in a table, or the part aligned
    temperature are not aligned.
    """
    def apply(self, part_span, current_span):
        """
        Returns True is the tuple passes, False if it should be throttled
        """
        # if both are in the same table
        if (part_span.parent.table is not None and current_span.parent.table is not None):
            if (part_span.parent.table == current_span.parent.table):
                return True

        # if part is in header, current is in table
        if (part_span.parent.table is None and current_span.parent.table is not None):
            ngrams = set(get_row_ngrams(current_span))
            if ('collector' in ngrams and 'current' in ngrams):
                return True

        # if neither part or temp is in table
        if (part_span.parent.table is None and current_span.parent.table is None):
            ngrams = set(get_phrase_ngrams(current_span))
            num_numbers = list(get_phrase_ngrams(current_span, attrib="ner_tags")).count('number')
            if ('collector' in ngrams and 'current' in ngrams and num_numbers <= 2):
                return True

        return False

    def aligned(self, span1, span2):
        ngrams = set(get_row_ngrams(span2))
        return  (span1.parent.table == span2.parent.table and
            (span1.parent.row_num == span2.parent.row_num or span1.parent.col_num == span2.parent.col_num))
