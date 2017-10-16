import re
from snorkel.lf_helpers import (
    get_left_tokens, get_right_tokens, get_between_tokens,
    get_text_between, get_tagged_text,
)

class PatternMatchFactory(object):
    """
    Simple class for generating LFs for binary relation classification
    """
    def __init__(self, name, search='sentence', window=1, attrib='words'):
        """

        :param name:
        :param search:
        :param window:
        :param attrib:
        """
        self.name = name
        self.search = search
        self.window = window
        self.attrib = attrib

    def _get_search_func(self, c):
        """
        Enumerate the token search space for pattern matching
        :param c:
        :return:
        """
        if self.search == "sentence":
            return c.get_parent().__dict__[self.attrib]
        elif self.search == "between":
            return get_text_between(c).strip().split()
        elif self.search == "left":
            # use left-most Span
            span = c[0] if c[0].char_start < c[1].char_start else c[1]
            return get_left_tokens(span, window=self.window, attrib=self.attrib)
        elif self.search == "right":
            # use right-most Span
            span = c[0] if c[0].char_start > c[1].char_start else c[1]
            return get_right_tokens(span, window=self.window, attrib=self.attrib)


class MatchTerms(PatternMatchFactory):
    """
    Match over several term dictionaries
    """
    def __init__(self, name, terms, label, search='sentence', window=1, attrib='words'):
        super(MatchTerms, self).__init__(name, search, window, attrib)
        self.terms = terms
        self.label = label

    def lf(self):
        def f(c):
            context = self._get_search_func(c)
            matches = self.terms.intersection(context)
            return self.label if len(matches) > 0 else 0

        params = "[{}|{}{}]".format(self.search, self.attrib, "" \
            if self.search in ['sentence', 'between'] \
            else "|window={}".format(self.window))

        args = ['LF_TERMS', self.name, params, 'TRUE' if self.label == 1 else 'FALSE']
        f.__name__ = "_".join(args)
        return f


class MatchRegex(PatternMatchFactory):
    """
    Match over a set of provided regular expressions
    """
    def __init__(self, name, rgxs, label, search='sentence', window=1, attrib='words'):
        super(MatchRegex, self).__init__(name, search, window, attrib)
        self.rgxs = [re.compile(pattern) for pattern in rgxs]
        self.label = label

    def lf(self):

        def f(c):
            context = ' '.join(self._get_search_func(c))
            # HACK: Just do a linear search for now. We could do something
            # smarter, like compile all regexes in the provided set into a DFA
            for rgx in self.rgxs:
                m = rgx.search(context)
                if m:
                    return self.label
            return 0

        params = "[{}|{}{}]".format(self.search, self.attrib, "" \
            if self.search in ['sentence', 'between'] \
            else "|window={}".format(self.window))

        args = ['LF_REGEX', self.name, params, 'TRUE' if self.label == 1 else 'FALSE']
        f.__name__ = "_".join(args)
        return f


class DistantSupervision(object):
    """
    Distant supervision for binary relations, representing facts as tuples
    """
    def __init__(self, name, kb, membership_func=None):
        self.name = name
        self.kb   = kb
        self.membership_func = membership_func if membership_func != None else DistantSupervision._exact_match

    @staticmethod
    def _exact_match(c, kb):
        s1, s2 = c[0].get_span(), c[1].get_span()
        return ((s1, s2) in kb or (s2, s1) in kb)

    def lf(self):
        def f(c):
            v = self.membership_func(c,self.kb)
            return 1 if v else 0

        args = ['LF_DIST_SUPERVISION', self.name, "TRUE"]
        f.__name__ = "_".join(args)
        return f