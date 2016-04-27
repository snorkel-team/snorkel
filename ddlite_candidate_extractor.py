import re

class CandidateExtractor(object):
  """
  Base class for candidate extraction
  A candidate extractor is composed from a set of CandidateExtractor child-class operators
  
  Calling apply(s) returns a list of lists of indices representing candidates
  """
  def __init__(self, *children, **opts):
    self.children = children
    self.opts     = opts

    # Global opts
    self.n                  = self.opts.get('n', 3)
    self.longest_match_only = self.opts.get('longest_match_only', True)
    self.attrib             = self.opts.get('attrib', 'words')
    self.label              = self.opts.get('label', None)

    # Load child opts
    self.init()

  def init(self):
    pass

  def _filter(self, idxs, seq):
    """This is the filter of the individual operator, which is then composed with other operators"""
    return True

  def filter(self, idxs, seq):
    """By default, operators are recursively composed with AND semantics"""
    if len(self.children) > 0:
      return self._filter(idxs, seq) and all([c.filter(idxs, seq) for c in self.children])
    else:
      return self._filter(idxs, seq)

  def apply(self, s):
    """Apply the candidate extractor to a Sentence object"""
    seq = s if isinstance(s, dict) else s.__dict__
    L   = len(seq[self.attrib])

    # Keep track of indexes we've already matched so that we can e.g. keep longest match only
    matched_seqs = []

    # Loop over all ngrams *in reverse order (longest first)*
    for l in range(1, self.n+1)[::-1]:
      for i in range(L-l+1):
        cidxs = range(i, i+l)

        # If we are only taking the longest match, skip if a subset of already-tagged idxs
        if self.longest_match_only and any(set(cidxs) <= ms for ms in matched_seqs):
          continue

        # Filter by filter
        if self.filter(cidxs, seq):
          matched_seqs.append(frozenset(cidxs))
          yield cidxs
  
  def print_apply(self, s):
    """Helper for debugging"""
    for cidxs in self.apply(s):
      print cidxs


class DictionaryMatch(CandidateExtractor):
  """Selects candidate ngrams that match against a given list d""" 
  def init(self):
    self.d           = frozenset(self.opts['d'])
    self.ignore_case = self.opts.get('ignore_case', True)
    self.sep         = self.opts.get('sep', ' ')
    self.attrib      = self.opts.get('attrib', 'words')

  def _filter(self, idxs, seq):
    s = seq[self.attrib]
    return self.sep.join(s[i] for i in idxs) in self.d


class Union(CandidateExtractor):
  """Selects the union of two or more candidate extractors"""
  def filter(self, idxs, seq):
    return any([c.filter(idxs, seq) for c in self.children])


class Concat(CandidateExtractor):
  """Selects candidates which are the concatenation of adjacent matches from child operators"""
  def filter(self, idxs, seq):
    if len(self.children) > 2:
      raise ValueError("More than two child CandidateExtractors not currently supported by Concat.")
    for split in range(1,len(idxs)):
      if self.children[0].filter(idxs[:split], seq) and self.children[1].filter(idxs[split:], seq):
        return True
    return False


class RegexMatch(CandidateExtractor):
  """Filters by regex pattern matching"""
  def init(self):
    self.semantics   = self.opts.get('semantics', 'concat')
    self.rgx         = self.opts['rgx']
    self.ignore_case = self.opts.get('ignore_case', True)
    self.rgx_comp    = re.compile(self.rgx, flags=re.I if self.ignore_case else 0)
    self.sep         = self.opts.get('sep', ' ')
    self.attrib      = self.opts.get('attrib', 'words')

  def _filter(self, idxs, seq):
    s = seq[self.attrib]
    if self.semantics.lower() == 'concat':
      return self.rgx_comp.match(self.sep.join(s[i] for i in idxs)) is not None
    elif self.semantics.lower() == 'all':
      return all([self.rgx_comp.match(s[i]) is not None for i in idxs])
    elif self.semantics.lower() == 'any':
      return any([self.rgx_comp.match(s[i]) is not None for i in idxs])


class TagMatch(RegexMatch):
  """Filters by matching tags (using rgx patterns), for e.g. POS tags"""
  def init(self):
    self.rgx       = self.opts['tag']
    self.rgx_comp  = re.compile(self.rgx, flags=re.I)
    self.semantics = 'all'
    self.attrib    = self.opts.get('attrib', 'poses')


class FuzzyDictionaryMatch(CandidateExtractor):
  """Matches against a dictionary with fuzzy phrase-level matching"""
  def _filter(self, idxs, seq):
    raise NotImplementedError()
