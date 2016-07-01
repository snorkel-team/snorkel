import re
from itertools import chain

class Matcher(object):
    """
    Applies a function f : c -> {0,1} to a generator of candidates,
    returning only candidates _c_ s.t. _f(c) == 1_,
    where f can be compositionally defined.
    """
    def __init__(self, *children, **opts):
        self.children = children
        self.opts     = opts
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
        if len(children) == 0:
            return self._f(c)
        elif len(children) == 1:
            return self._f(c) * self.children[0].f(c)
        else:
            raise Exception("%s does not support more than one child Matcher" % self.__name__)

    def apply(self, candidates):
        """Apply the Matcher to a **generator** of candidates"""
        for c in candidates:
            if self.f(c) > 0:
                yield c


class DictionaryMatch(Matcher):
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
