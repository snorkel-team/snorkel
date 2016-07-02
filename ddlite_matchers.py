import re
from itertools import chain

class Matcher(object):
    """
    Applies a function f : c -> {0,1} to a generator of candidates,
    returning only candidates _c_ s.t. _f(c) == 1_,
    where f can be compositionally defined.
    """
    def __init__(self, *children, **opts):
        self.children           = children
        self.opts               = opts
        self.longest_match_only = self.opts.get('longest_match_only', False)
        self.init()
    
    def init(self):
        pass

    def _f(self, c):
        """The internal (non-composed) version of filter function f"""
        return 1

    def f(self, c):
        """
        The recursicvely composed version of filter function f
        By default, returns logical **conjunction** of opeerator and single child operator
        """
        if len(self.children) == 0:
            return self._f(c)
        elif len(self.children) == 1:
            return self._f(c) * self.children[0].f(c)
        else:
            raise Exception("%s does not support more than one child Matcher" % self.__name__)

    def _is_subspan(self, c, span):
        """Tests if candidate c is subspan of span, where span is defined specific to candidate type"""
        return False

    def _get_span(self, c):
        """Gets a tuple that identifies a span for the specific candidate class that c belongs to"""
        return c

    def apply(self, candidates):
        """
        Apply the Matcher to a **generator** of candidates
        Optionally only takes the longest match (NOTE: assumes this is the *first* match)
        """
        seen_spans = set()
        for c in candidates:
            if self.f(c) > 0 and (not self.longest_match_only or not any([self._is_subspan(c, s) for s in seen_spans])):
                if self.longest_match_only:
                    seen_spans.add(self._get_span(c))
                yield c


class NgramMatcher(Matcher):
    """Matcher base class for Ngram objects"""
    def _is_subspan(self, c, span):
        """Tests if candidate c is subspan of span, where span is defined specific to candidate type"""
        return c.char_start >= span[0] and c.char_end <= span[1]

    def _get_span(self, c):
        """Gets a tuple that identifies a span for the specific candidate class that c belongs to"""
        return (c.char_start, c.char_end)


class DictionaryMatch(NgramMatcher):
    """Selects candidate Ngrams that match against a given list d"""
    def init(self):
        self.d           = frozenset(self.opts['d'])
        self.ignore_case = self.opts.get('ignore_case', True) 
        self.attrib      = self.opts.get('attrib', 'words')
    
    def _f(self, c):
        return 1 if c.get_attrib_span(self.attrib) in self.d else 0


class Union(Matcher):
    """Takes the union of candidate sets returned by child operators"""
    def f(self, c):
       for child in self.children:
           if child.f(c) > 0:
               return 1
       return 0
