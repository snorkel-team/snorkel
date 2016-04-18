import bisect, re
from collections import defaultdict

class CandidateExtractor(object):
  """
  A CandidateExtractor object takes as input:
    * Optionally, one or more other CandidateExtractor objects (it is *compositional*)
    * A list of keyword arguments
  and can then be *applied* to a context object to extract candidates from it
  """
  def __init__(self, *candidate_extractors, **opts):
    self.candidate_extractors = candidate_extractors
    if len(self.candidate_extractors) > 0:
      # NOTE: using issubclass is problematic here...
      if not all([hasattr(c, 'apply') for c in self.candidate_extractors]):
        raise ValueError("Non-keyword args must be CandidateExtractor subclass.")
    self.opts = opts
    self.init()

  def init(self):
    """This function initializes all CandidateExtractor subclass-specific keyword args"""
    raise NotImplementedError()

  def apply(self, s):
    """
    A CandidateExtractor is *applied* to a context s using this function, where s is a list, e.g.
    of words or lemmas
    The output is a generator of (indexes,label) pairs which represent extracted candidates
    """
    # Base case: call subclass-internal _apply function
    if len(self.candidate_extractors) == 0:
      for idxs,label in self._apply(s):
        yield idxs,label

    # If defined compositionally, recurse
    # NOTE: this implicitly unions multiple input candidate extractors
    else:
      for ce in self.candidate_extractors:
        for idxs,label in ce.apply(s):
          for idxs_out,label_out in self._apply(s, idxs=idxs):
            l = '%s : %s' % (label_out, label) if len(label_out) > 0 else label
            yield idxs_out,l
  
  def _apply(self, s, idxs=None):
    """This function defines the actual extraction operation done in apply for a subclass"""
    raise NotImplementedError()

  def _get_attrib_seq(self, s):
    """Helper util to get the match attrib of the input context"""
    # Make sure we're operating on a dict, then get match_attrib
    try:
      return s[self.match_attrib]
    except TypeError:
      return s.__dict__[self.match_attrib]


class Union(CandidateExtractor):
  """Takes the union of two or more candidate extractors"""
  def init(self):
    if len(self.candidate_extractors) == 0:
      raise ValueError("Union must have one or more CandidateExtractors as inputs.")

  def _apply(self, s, idxs):
    yield idxs, ''

    
class DictionaryMatch(CandidateExtractor):
  """Selects according to ngram-matching against a dictionary i.e. list of words"""
  def init(self):
    # Load opts- this is from the kwargs dict
    self.label         = self.opts['label']
    self.dictionary    = self.opts['dictionary']
    self.match_attrib  = self.opts.get('match_attrib', 'words')
    self.ignore_case   = self.opts.get('ignore_case', True)
    self.longest_match = self.opts.get('longest_match', True)

    # Split the dictionary up by phrase length (i.e. # of tokens)
    self.dl = defaultdict(lambda : set())
    for phrase in self.dictionary:
      self.dl[len(phrase.split())].add(phrase.lower() if self.ignore_case else phrase)
    self.dl.update((k, frozenset(v)) for k,v in self.dl.iteritems())

    # Get the *DESC order* ngram range for this dictionary
    self.ngr = range(max(1, min(self.dl.keys())), max(self.dl.keys())+1)[::-1]

  def _apply(self, s, idxs=None):
    """
    Take in an object or dictionary which contains match_attrib
    and get the index lists of matching phrases
    If idxs=None, consider all indices, otherwise constrain to subset of idxs
    """
    seq = self._get_attrib_seq(s)

    # If idxs=None, consider the full index range, otherwise only subseqs of idxs
    start = 0 if idxs is None else min(idxs)
    end   = len(seq) if idxs is None else max(idxs)+1
    L     = len(seq) if idxs is None else len(idxs)

    # NOTE: We assume that idxs is a range of consecutive indices!
    if L != end - start:
      raise ValueError("Candidates must be over consecutive spans of indices")

    # Keep track of indexes we've already matched so that we can e.g. keep longest match only
    matched_seqs = []

    # Loop over all ngrams
    for l in filter(lambda n : n <= L, self.ngr):
      for i in range(start, end-l+1):
        ssidx = range(i, i+l)

        # If we are only taking the longest match, skip if a subset of already-tagged idxs
        if self.longest_match and any(set(ssidx) <= ms for ms in matched_seqs):
          continue
        phrase = ' '.join(seq[i:i+l])
        phrase = phrase.lower() if self.ignore_case else phrase
        if phrase in self.dl[l]:
          matched_seqs.append(frozenset(ssidx))
          yield list(ssidx), self.label


class RegexBase(CandidateExtractor):
  """Parent class for Regex-related candidate extractors"""
  def init(self):
    self.label        = self.opts['label']
    self.match_attrib = self.opts.get('match_attrib', 'words')
    regex_pattern     = self.opts['regex_pattern']
    ignore_case       = self.opts.get('ignore_case', True)
    self._re_comp     = re.compile(regex_pattern, flags=re.I if ignore_case else 0)
    self.sep          = self.opts.get('sep', ' ')


class RegexFilterAny(RegexBase):
  """Filter candidates where *any* element matches the regex"""
  def _apply(self, s, idxs=None):
    seq = self._get_attrib_seq(s)
    idxs = range(len(seq)) if idxs is None else idxs
    if any(self._re_comp.match(seq[i]) is not None for i in idxs):
      yield idxs, self.label


class RegexFilterAll(RegexBase):
  """Filter candidates where *all* elements match the regex"""
  def _apply(self, s, idxs=None):
    seq = self._get_attrib_seq(s)
    idxs = range(len(seq)) if idxs is None else idxs
    if all(self._re_comp.match(seq[i]) is not None for i in idxs):
      yield idxs, self.label


class RegexFilterConcat(RegexBase):
  """Filter candidates where the concatenated elements phrase matches the regex"""
  def _apply(self, s, idxs=None):
    seq = self._get_attrib_seq(s)
    idxs = range(len(seq)) if idxs is None else idxs
    if self._re_comp.match(self.sep.join(seq[i] for i in idxs)) is not None:
      yield idxs, self.label


class RegexNgramMatch(RegexBase):
  """Selects according to ngram-matching against a regex """
  def _apply(self, s, idxs=None):
    seq = self._get_attrib_seq(s)
    idxs = range(len(seq)) if idxs is None else idxs
    idx_set = set(idxs)
    # Convert character index to token index and form phrase
    if self.match_attrib == 'text':
      try:
        start_c_idx = s['token_idxs']
      except TypeError:
        start_c_idx = s.__dict__['token_idxs']
      start_c_idx = [c - start_c_idx[0] for c in start_c_idx]
    else:
      start_c_idx = [0]
      for s in seq:
        start_c_idx.append(start_c_idx[-1] + len(s) + 1)
    # Find regex matches over phrase
    is_txt = (self.match_attrib == 'text')
    phrase = seq if is_txt else self.sep.join(seq[i] for i in idxs)
    for match in self._re_comp.finditer(phrase):
      start = bisect.bisect(start_c_idx, match.start())
      end = bisect.bisect(start_c_idx, match.end())
      rg = list(range(start-1, end))
      if not is_txt:
        yield [idxs[i] for i in rg], self.label 
      elif set(rg).issubset(idx_set):
        yield rg, self.label

