import os
from collections import namedtuple, defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.parse.stanford import StanfordDependencyParser
from nltk.internals import find_jars_within_path
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from tree_structs import corenlp_to_xmltree
from treedlib import compile_relation_feature_generator

Sentence = namedtuple('Sentence', 'words, lemmas, poses, dep_parents, dep_labels')

class SentenceParser:
  def __init__(self):

    # Init Porter stemmer
    self.stemmer = PorterStemmer()

    # A hackey fix is needed here to load the Stanford parser correctly
    # Ref: https://gist.github.com/alvations/e1df0ba227e542955a8a
    PARSER = "%s/parser" % os.getcwd()
    p = '$CLASSPATH:{0}:{0}/stanford-parser.jar:{0}/stanford-parser-3.6.0-models.jar'.format(PARSER)
    os.environ["CLASSPATH"] = p
    self.parser = StanfordDependencyParser()
    stanford_dir = self.parser._classpath[0].rpartition('/')[0]
    self.parser._classpath = tuple(find_jars_within_path(stanford_dir))

  def _parse_sent(self, words, conll):
    """Parse a single sentence- input as a CONLL-4 array- returning a Sentence object"""
    toks, poses, dep_parents, dep_labels = zip(*filter(lambda l : len(l) == 4, conll))

    # Correct the indexing error introduced by dropping punctuation w/out adjusting idxs
    # TODO: Find the damn flag that keeps punctuation!
    offsets = []
    d = 0
    for i in range(len(words)):
      if i+d >= len(toks) or words[i] != toks[i+d]:
        d -= 1
      offsets.append(d)

    return Sentence(
      words=list(toks),
      lemmas=[self.stemmer.stem(t.lower()) for t in toks],
      poses=list(poses),
      dep_parents=[i + offsets[i-1] if i > 0 else 0 for i in map(int, dep_parents)],
      dep_labels=list(dep_labels))

  def parse(self, doc):
    """Parse a raw document as a string into a list of sentences"""
    # Split into sentences & words
    sents = map(word_tokenize, sent_tokenize(doc))

    # Pass in all sents to parser- note there is a performance gain here
    # that could surely be exploited more...
    for i,parse in enumerate(self.parser.parse_sents(sents)):
      conll = [l.split('\t') for l in list(parse)[0].to_conll(4).split('\n')]
      yield self._parse_sent(sents[i], conll)   


class DictionaryMatch:
  """Selects according to ngram-matching against a dictionary i.e. list of words"""
  def __init__(self, label, dictionary, match_attrib='words', ignore_case=True):
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
          yield list(range(i, i+l))


def tag_seq(words, seq, tag):
  """Sub in a tag for a subsequence of a list"""
  words_out = words[:seq[0]] + ['{{%s}}' % tag]
  words_out += words[seq[-1] + 1:] if seq[-1] < len(words) - 1 else []
  return words_out

def tag_seqs(words, seqs, tags):
  """
  Given a list of words, a *list* of lists of indexes, anmd the corresponding tags
  This function substitutes the tags for the words coresponding to the index lists,
  taking care of shifting indexes appropriately after multi-word substitutions
  NOTE: this assumes non-overlapping seqs!
  """
  words_out = words
  dj = 0
  for i in np.argsort(seqs):
    i = int(i)
    words_out = tag_seq(words_out, map(lambda j : j - dj, seqs[i]), tags[i])
    dj += len(seqs[i]) - 1
  return words_out


class Relation:
  def __init__(self, e1_idxs, e2_idxs, e1_label, e2_label, sent, xt):
    self.e1_idxs = e1_idxs
    self.e2_idxs = e2_idxs
    self.idxs = [self.e1_idxs, self.e2_idxs]
    self.e1_label = e1_label
    self.e2_label = e2_label
    self.labels = [self.e1_label, self.e2_label]

    # Absorb XMLTree and Sentence object attributes for access by rules
    self.xt = xt
    self.root = self.xt.root
    self.__dict__.update(sent.__dict__)

    # Add some additional useful attibutes
    self.tagged_sent = ' '.join(tag_seqs(self.words, self.idxs, self.labels))

  def render(self):
    self.xt.render_tree(self.idxs)

  def __repr__(self):
    return '<Relation: %s - %s>' % (self.e1_idxs, self.e2_idxs)


class Relations:
  def __init__(self, e1, e2, sents):
    self.e1 = e1
    self.e2 = e2
    self.relations = list(self.extract(sents))
    self.X = None
    self.F = None
    self.feat_index = {}

  def extract(self, sents):
    for sent in sents:
      for rel in self._apply(sent):
        yield rel

  def _apply(self, sent):
    xt = corenlp_to_xmltree(sent)
    for e1_idxs in self.e1.apply(sent):
      for e2_idxs in self.e2.apply(sent):
        yield Relation(e1_idxs, e2_idxs, self.e1.label, self.e2.label, sent, xt)

  def apply_rules(self, rules):
    self.X = np.zeros((len(rules), len(self.relations)))
    for i,rule in enumerate(rules):
      for j,rel in enumerate(self.relations):
        self.X[i,j] = rule(rel)

  def extract_features(self, method='treedlib'):
    get_feats = compile_relation_feature_generator()
    f_index = defaultdict(list)
    for j,rel in enumerate(self.relations):
      for feat in get_feats(rel.root, rel.e1_idxs, rel.e2_idxs):
        f_index[feat].append(j)

    # Apply the feature generator, constructing a sparse matrix incrementally
    # Note that lil_matrix should be relatively efficient as we proceed row-wise
    self.F = lil_matrix((len(f_index), len(self.relations)))
    for i,feat in enumerate(f_index.keys()):
      self.feat_index[i] = feat
      for j in f_index[feat]:
        self.F[i,j] = 1
    self.F = csr_matrix(self.F)
