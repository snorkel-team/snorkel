from itertools import chain
import re

# op_0 : X -> p(T)
# op_1 : p(T) -> p(T)
# ind : p(T) -> {0,1}^F

class FeatureTemplate:
  """Base feature template class"""
  def __init__(self):
    self.label = None
    self.xpaths = set(['//node'])
    self.subsets = None  # subtrees / p(T)

  def apply(self, root):
    """
    Applies the feature template to the XML tree (the root of which is provided as input)
    Generates string feature representations
    Relies on a generator subfunction, _get_result_sets,  which returns result sets of nodes, 
    then optionally gets subsets (e.g. for n-grams)
    """
    for res in self._get_result_sets(root):
      if len(res) > 0:
        res_sets = [res] if self.subsets is None else subsets(res, self.subsets)
        for res_set in res_sets:
          yield self._feat_str(res_set)

  def _get_result_sets(self, root):
    """Default result set generator- just applies each xpath to the root node"""
    for xpath in self.xpaths:
      yield root.xpath(xpath)

  def _feat_str(self, res_set):
    return '%s[%s]' % (self.label, '_'.join(map(str, res_set)))

  def apply_and_print(self, root):
    """Helper function to apply and then print the features one per line"""
    for f in self.apply(root):
      print f

  def __repr__(self):
    return "<%s, XPaths='%s', subsets=%s>" % (self.label, self.xpaths, self.subsets)


def subsets(x, L):
  """Return all subsets of length 1, 2, ..., min(l, len(x)) from x"""
  return chain.from_iterable([x[s:s+l+1] for s in range(len(x)-l)] for l in range(min(len(x),L)))


class Mention(FeatureTemplate):
  """The feature comprising the set of nodes making up the mention"""
  def __init__(self, cid, subsets=None):
    self.label = 'MENTION'
    self.xpaths = set(["//node[@cid='%s']" % str(cid)])
    self.subsets = subsets


class Indicator(FeatureTemplate):
  """
  Outermost indicator feature, which just maps a specific attribute onto the inputted
  feature template and uses that feature template's apply function
  """
  def __init__(self, f, attrib):
    self.f = f
    self.label = '%s-%s' % (attrib.upper(), f.label)
    self.xpaths = set('%s/@%s' % (xpath, attrib) for xpath in f.xpaths)
    self.subsets = f.subsets

  def apply(self, root):
    self.f.label = self.label
    self.f.xpaths = self.xpaths
    return self.f.apply(root)


class Left(FeatureTemplate):
  """
  The feature comprising the set of *sibling* nodes to the left of the input feature's nodes
  """
  def __init__(self, f, subsets=3):
    self.label = 'LEFT-OF-%s' % f.label
    self.xpaths = set(xpath + '/preceding-sibling::node' for xpath in f.xpaths)
    self.subsets = subsets

  def _get_result_sets(self, root):
    """Only take the N nodes to the left, where N is = self.subsets"""
    for xpath in self.xpaths:
      yield root.xpath(xpath)[::-1][:self.subsets][::-1]


class Right(FeatureTemplate):
  """
  The feature comprising the set of *sibling* nodes to the right of the input feature's nodes
  """
  def __init__(self, f, subsets=3):
    self.label = 'RIGHT-OF-%s' % f.label
    self.xpaths = set(xpath + '/following-sibling::node' for xpath in f.xpaths)
    self.subsets = subsets

  def _get_result_sets(self, root):
    """Only take the N nodes to the right, where N is = self.subsets"""
    for xpath in self.xpaths:
      yield root.xpath(xpath)[:self.subsets]


class Between(FeatureTemplate):
  """
  The set of nodes *between* two node sets
  """
  def __init__(self, f1, f2, subsets=None):
    self.label = 'BETWEEN-%s-%s' % (f1.label, f2.label)
    self.xpaths = set(['/ancestor-or-self::node'])
    self.xpaths1 = f1.xpaths
    self.xpaths2 = f2.xpaths
    self.subsets = subsets

  def _get_result_sets(self, root):
    """
    Get the path between the two node sets by getting the lowest shared parent,
    then concatenating the two ancestor paths at this shared parent
    """
    for xpath in self.xpaths:
      for xpath1 in self.xpaths1:
        for xpath2 in self.xpaths2:
          p1 = root.xpath(xpath1 + xpath)
          p2 = root.xpath(xpath2 + xpath)
          shared = set(p1).intersection(p2)
          b1 = []
          b2 = []
          for node in reversed(p1):
            b1.append(node)
            if node in shared: break
          for node in reversed(p2):
            if node in shared: break
            b2.append(node)

          # Return only the path *between*, filtering out the self nodes
          # NOTE: This is different for node vs. edge attributes...
          res = b1 + b2[::-1]
          if 'dep_label' in xpath:
            yield res
          else:
            yield filter(lambda n : (len(p1)==0 or n!=p1[-1]) and (len(p2)==0 or n!=p2[-1]), res)


# TODO: Only handles single-word keywords right now!!!
class Keyword(FeatureTemplate):
  """Searches for matches against a dictionary (list) of keywords"""
  def __init__(self, kws, f=None):
    self.label = 'KEYWORD' if f is None else 'KEYWORD-IN-%s' % f.label
    xs = ['//node'] if f is None else f.xpaths
    self.xpaths = set(chain.from_iterable(["%s[@word='%s']" % (x,w) for w in kws] for x in xs))
    self.subsets = None


# Should these classes be semantically separated, into NodeSet and FeatureIndicator,
# Where some composition of FeatureIndicator classes are applied to some composition of
# NodeSet classes...?


# TODO: Handle multi-word properly!!  Or just scrap this, do rgx in python...
class RgxIndicator(FeatureTemplate):
  """Indicator of whether a regex matches the node set"""
  def __init__(self, rgx, attrib, label, f):
    self.label = label
    self.xpaths = set("%s/@%s[re:test(., '%s')]" % (xpath, attrib, rgx) for xpath in f.xpaths)
    self.subsets = None

  def apply(self, root):
    for xpath in self.xpaths:
      if len(root.xpath(xpath, namespaces={'re':"http://exslt.org/regular-expressions"})) > 0:
        yield self.label.upper()
