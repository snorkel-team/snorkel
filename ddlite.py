import os, sys
from collections import namedtuple, defaultdict
import random
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from tree_structs import corenlp_to_xmltree

sys.path.append('%s/treedlib' % os.getcwd())
from treedlib import compile_relation_feature_generator

from parser import Sentence, SentenceParser


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
    self.relations = list(self._extract(sents))
    self.rules = None
    self.feats = None
    self.X = None
    self.feat_index = {}
    self.w = None

  def _extract(self, sents):
    for sent in sents:
      for rel in self._apply(sent):
        yield rel

  def _apply(self, sent):
    xt = corenlp_to_xmltree(sent)
    for e1_idxs in self.e1.apply(sent):
      for e2_idxs in self.e2.apply(sent):
        yield Relation(e1_idxs, e2_idxs, self.e1.label, self.e2.label, sent, xt)

  def apply_rules(self, rules):
    self.rules = np.zeros((len(rules), len(self.relations)))
    for i,rule in enumerate(rules):
      for j,rel in enumerate(self.relations):
        self.rules[i,j] = rule(rel)

  def extract_features(self, method='treedlib'):
    get_feats = compile_relation_feature_generator()
    f_index = defaultdict(list)
    for j,rel in enumerate(self.relations):
      for feat in get_feats(rel.root, rel.e1_idxs, rel.e2_idxs):
        f_index[feat].append(j)

    # Apply the feature generator, constructing a sparse matrix incrementally
    # Note that lil_matrix should be relatively efficient as we proceed row-wise
    F = lil_matrix((len(f_index), len(self.relations)))
    for i,feat in enumerate(f_index.keys()):
      self.feat_index[i] = feat
      for j in f_index[feat]:
        F[i,j] = 1
    self.feats = csr_matrix(F)

  def learn_feats_and_weights(self, nSteps=1000, sample=False, nSamples=100, mu=1e-9, verbose=False):
    """
    Uses the R x N matrix of rules and the F x N matrix of features defined
    for the Relations object
    Stacks them, giving the rules a +1 prior (i.e. init value)
    Then runs learning, saving the learned weights
    """
    R, N = self.rules.shape  # dense
    F, N = self.feats.shape  # sparse
    self.X = np.array(np.vstack([self.rules, self.feats.todense()]))
    w0 = np.concatenate([np.ones(R), np.zeros(F)])
    self.w = learn_params(self.X, nSteps=nSteps, w0=w0, sample=sample, nSamples=nSamples, mu=mu, verbose=verbose)

  def get_predicted(self):
    """Get the array of predicted (boolean) variables given learned weight param w"""
    return np.sign(np.dot(self.X.T, self.w))

  def get_classification_accuracy(self, ground_truth):
    """Given the labels for the Relations set, return the classification accuracy"""
    if len(ground_truth) != self.X.shape[1]:
      raise ValueError("%s ground truth labels for %s relations." % (len(ground_truth), self.X.shape[1]))
    pred = self.get_predicted()
    return (np.dot(pred, ground_truth) / len(ground_truth) + 1) / 2

  def get_rule_priority_vote_accuracy(self, ground_truth):
    """
    This is to answer the question: 'How well would my rules alone do?'
    I.e. without any features, learning of rule or feature weights, etc.- this serves as a
    natural baseline / quick metric
    Labels are assigned by the first rule that emits one for each relation (based on the order
    of the provided rules list)
    """
    R, N = self.rules.shape
    if len(ground_truth) != N:
      raise ValueError("%s ground truth labels for %s relations." % (len(ground_truth), self.X.shape[1]))
    correct = 0
    for j in range(N):
      for i in range(R):
        if self.rules[i,j] != 0:
          correct += 1 if self.rules[i,j] == ground_truth[j] else 0
          break
    return float(correct) / N


#
# Logistic regression algs
# Ported from Chris's Julia notebook...
#
def log_odds(p):
  """This is the logit function"""
  return np.log(float(p) / (1-p))

def odds_to_prob(l):
  """
  This is the inverse logit function logit^{-1}:

    l       = \log\frac{p}{1-p}
    \exp(l) = \frac{p}{1-p}
    p       = \frac{\exp(l)}{1 + \exp(l)}
  """
  return np.exp(l) / (1.0 + np.exp(l))

def sample_data(X, w, nSamples):
  """
  Here we do Gibbs sampling over the decision variables (representing our objects), o_j
  corresponding to the columns of X
  The model is just logistic regression, e.g.

    P(o_j=1 | X_{*,j}; w) = logit^{-1}(w \dot X_{*,j})

  This can be calculated exactly, so this is essentially a noisy version of the exact calc...
  """
  if type(X) != np.ndarray or type(w) != np.ndarray:
    raise TypeError("Inputs should be np.ndarray type.")
  R, N = X.shape
  if w.shape != (R,):
    raise Exception("w should be an array of length %s" % R)
  t = np.zeros(N)
  f = np.zeros(N)

  # Take samples of random variables
  for i in range(nSamples):
    idx = random.randint(0, N-1)
    if random.random() < odds_to_prob(np.dot(X[:,idx].T, w)):
      t[idx] += 1
    else:
      f[idx] += 1
  return t, f

def exact_data(X, w):
  """
  We calculate the exact conditional probability of the decision variables in
  logistic regression; see sample_data
  """
  if type(X) != np.ndarray or type(w) != np.ndarray:
    raise TypeError("Inputs should be np.ndarray type.")
  R, N = X.shape
  if w.shape != (R,):
    raise Exception("w should be an array of length %s" % R)
  t = np.array(map(odds_to_prob, np.dot(X.T, w)))
  return t, 1-t

def transform_sample_stats(X, t, f):
  """
  Here we calculate the expected accuracy of each rule/feature
  (corresponding to the rows of X) wrt to the distribution of samples S:

    E_S[ accuracy_i ] = E_(t,f)[ \frac{TP + TN}{TP + FP + TN + FN} ]
                      = \frac{X_{i|x_{ij}>0}*t - X_{i|x_{ij}<0}*f}{t+f}
                      = \frac12\left(\frac{X*(t-f)}{t+f} + 1\right)
  """
  if type(X) != np.ndarray:
    raise TypeError("Inputs should be np.ndarray type.")
  n_pred = np.diag(np.dot(abs(X), t+f))
  p_correct = (np.diag(np.linalg.inv(n_pred)*(np.dot(X, t) - np.dot(X, f))) + 1) / 2
  return p_correct, np.diag(n_pred)

def learn_params(X, nSteps, w0=None, sample=True, nSamples=100, mu=1e-9, verbose=False):
  """We perform SGD wrt the weights w"""
  if type(X) != np.ndarray:
    raise TypeError("Inputs should be np.ndarray type.")
  R, N = X.shape

  # We initialize w at 1 for rules & 0 for features
  # As a default though, if no w0 provided, we initialize to all zeros
  w = np.zeros(R) if w0 is None else w0
  g = np.zeros(R)
  l = np.zeros(R)

  # Take SGD steps
  for step in range(nSteps):
    if step % 100 == 0 and verbose:
      print "Learning epoch = %s" % step

    # Get the expected rule accuracy
    t,f = sample_data(X, w, nSamples=nSamples) if sample else exact_data(X, w)
    p_correct, n_pred = transform_sample_stats(X, t, f)

    # Get the "empirical log odds"; NB: this assumes one is correct, clamp is for sampling...
    l = np.clip(map(log_odds, p_correct), -10, 10)

    # SGD step, with \ell_2 regularization, and normalization by the number of samples
    g0 = (n_pred*(w - l)) / np.sum(n_pred) + mu*w

    # Momentum term for faster training
    g = 0.95*g0 + 0.05*g

    # Update weights
    w -= 0.01*g
  return w
