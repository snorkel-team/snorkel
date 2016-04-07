import bisect, re, warnings
from collections import defaultdict

class Matcher(object):
  def apply(self, s):
    raise NotImplementedError()
    
class DictionaryMatch(Matcher):
  """Selects according to ngram-matching against a dictionary i.e. list of words"""
  def __init__(self, label, dictionary, match_attrib='words', ignore_case=True):
    if match_attrib in ['sent_id', 'doc_id', 'text', 'token_idxs']:
      raise ValueError("Match attribute cannot be ID or sentence text")
    self.label = label
    self.match_attrib = match_attrib
    self.ignore_case = ignore_case

    # Split the dictionary up by phrase length (i.e. # of tokens)
    self.dl = defaultdict(lambda : set())
    for phrase in dictionary:
      self.dl[len(phrase.split())].add(phrase.lower() if ignore_case else phrase)
    self.dl.update((k, frozenset(v)) for k,v in self.dl.iteritems())

    # Get the ngram range for this dictionary
    self.ngr = range(max(1, min(self.dl.keys())), max(self.dl.keys())+1)

  def apply(self, s):
    """
    Take in an object or dictionary which contains match_attrib
    and get the index lists of matching phrases
    """
    # Make sure we're operating on a dict, then get match_attrib
    try:
      seq = s[self.match_attrib]
    except TypeError:
      seq = s.__dict__[self.match_attrib]

    # Loop over all ngrams
    for l in self.ngr:
      for i in range(0, len(seq)-l+1):
        phrase = ' '.join(seq[i:i+l])
        phrase = phrase.lower() if self.ignore_case else phrase
        if phrase in self.dl[l]:
          yield list(range(i, i+l)), self.label

class RegexMatch(Matcher):
  """Selects according to ngram-matching against a regex """
  def __init__(self, label, regex_pattern, match_attrib='words', ignore_case=True):
    if match_attrib in ['sent_id', 'doc_id', 'token_idxs']:
      raise ValueError("Match attribute cannot be ID")
    self.label = label
    self.match_attrib = match_attrib
    # Ignore whitespace when we join the match attribute
    self._re_comp = re.compile(regex_pattern, flags=re.I if ignore_case else 0)

  def apply(self, s):
    """
    Take in an object or dictionary which contains match_attrib
    and get the index lists of matching phrases
    """
    # Make sure we're operating on a dict, then get match_attrib
    try:
      seq = s[self.match_attrib]
      start_c_idx = s['token_idxs']
    except TypeError:
      seq = s.__dict__[self.match_attrib]
      start_c_idx = s.__dict__['token_idxs']
    
    # Convert character index to token index and form phrase
    if self.match_attrib == 'text':
      start_c_idx = [c - start_c_idx[0] for c in start_c_idx]
    else:
      start_c_idx = [0]
      for s in seq:
        start_c_idx.append(start_c_idx[-1] + len(s) + 1)
    # Find regex matches over phrase
    phrase = seq if self.match_attrib == 'text' else ' '.join(seq)
    for match in self._re_comp.finditer(phrase):
      start = bisect.bisect(start_c_idx, match.start())
      end = bisect.bisect(start_c_idx, match.end())
      yield list(range(start-1, end)), self.label
      
class MultiMatcher(Matcher):
  """ 
  Wrapper to apply multiple matchers of a given entity type 
  Priority of labeling given by matcher order
  """
  def __init__(self, *matchers, **kwargs):
    if len(matchers) > 0:
      [warnings.warn("Non-Matcher object passed to MultiMatcher")
       for m in matchers if not issubclass(m.__class__, Matcher)]
      self.matchers = matchers
    else:
      raise ValueError("Need at least one matcher")
    self.label = kwargs['label'] if 'label' in kwargs else None
    
  def apply(self, s):
    applied = set()
    for m in self.matchers:
      for rg, m_label in m.apply(s):
        rg_end = (rg[0], rg[-1])
        if rg_end not in applied:
          applied.add(rg_end)
          yield rg, self.label if self.label is not None else m_label

def main():
  from ddlite import SentenceParser
  txt = "Han likes Luke and a good-wookie. Han Solo don\'t like 88-IG."
  parser = SentenceParser()
  sents = list(parser.parse(txt))

  g = DictionaryMatch('G', ['Han Solo', 'Luke', 'wookie'])
  b1 = RegexMatch('B1', "\d+", match_attrib="text")
  b2 = RegexMatch('B2', "\d+", match_attrib="words")
  
  print list(g.apply(sents[0])) 
  print list(b1.apply(sents[1]))
  print list(b2.apply(sents[1]))  

if __name__ == '__main__':
  main()
  