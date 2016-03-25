# Base Python
import cPickle, json, os, sys, warnings
from collections import defaultdict
import lxml.etree as et

# Scientific modules
import numpy as np
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", module="matplotlib")
import matplotlib.pyplot as plt
import scipy.sparse as sparse

# Feature modules
sys.path.append('{}/treedlib'.format(os.getcwd()))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree, XMLTree
from ddlite_entity_features import *

# ddlite parsers
from ddlite_parser import *

# ddlite matchers
from ddlite_matcher import *

# ddlite mindtagger
from ddlite_mindtagger import *

#####################################################################
############################ TAGGING UTILS ##########################
#####################################################################

def tag_seq(words, seq, tag):
  """Sub in a tag for a subsequence of a list"""
  words_out = words[:seq[0]] + ['{{%s}}' % tag]
  words_out += words[seq[-1] + 1:] if seq[-1] < len(words) - 1 else []
  return words_out

def tag_seqs(words, seqs, tags):
  """
  Given a list of words, a *list* of lists of indexes, and the corresponding tags
  This function substitutes the tags for the words coresponding to the index lists,
  taking care of shifting indexes appropriately after multi-word substitutions
  NOTE: this assumes non-overlapping seqs!
  """
  words_out = words
  dj = 0
  for i in np.argsort(seqs, axis=0):
    i = int(i[0]) if hasattr(i, '__iter__') else int(i)
    words_out = tag_seq(words_out, map(lambda j : j - dj, seqs[i]), tags[i])
    dj += len(seqs[i]) - 1
  return words_out

#####################################################################
############################ EXTRACTIONS ############################
#####################################################################

class Extraction(object):
  """ Proxy providing an interface into the Extractions class """
  def __init__(self, extractions, ex_id, p=None):
    self.E = extractions
    self.id = ex_id
    self._p = p
  def __getattr__(self, name):
    if name.startswith('prob'):
      return self._p
    return getattr(self.E._extractions[self.id], name)
  def __repr__(self):
    s = str(self.E._extractions[self.id])
    return s if self._p is None else (s + " with probability " + str(self._p))

class extraction_internal(object):
  """
  Base class for an extraction
  See entity_internal and relation_internal for examples
  """
  def __init__(self, all_idxs, labels, sent, xt):
    self.all_idxs = all_idxs
    self.labels = labels
    # Absorb XMLTree and Sentence object attributes for access by rules
    self.xt = xt
    self.root = self.xt.root
    self.__dict__.update(sent.__dict__)

    # Add some additional useful attributes
    self.tagged_sent = ' '.join(tag_seqs(self.words, self.all_idxs, self.labels))

  def render(self):
    self.xt.render_tree(self.all_idxs)
  
  def __getstate__(self):
    cp = self.__dict__.copy()
    del cp['root']
    cp['xt'] = cp['xt'].to_str()
    return cp
    
  def __setstate__(self, d):
    self.__dict__ = d
    self.xt = XMLTree(et.fromstring(d['xt']), self.words)
    self.root = self.xt.root    
  
  def __repr__(self):
    raise NotImplementedError()
    
class Extractions(object):
  """
  Base class for a collection of extractions
  Sub-classes need to yield extractions from sentences (_apply) and 
  generate features (_get_features)
  See Relations and Entities for examples
  """
  def __init__(self, C):
    """
    Set up learning problem and generate candidates
     * C is a flat list of Sentence objects or a path to pickled extractions
    """
    if isinstance(C, basestring):
      try:
        with open(C, 'rb') as f:
          self._extractions = cPickle.load(f)
      except:
        raise ValueError("No pickled extractions at {}".format(C))
    else:
      self._extractions = list(self._extract(C))
    self.feats = None
    self.feat_index = {}
  
  def __getitem__(self, i):
    return Extraction(self, i)
    
  def __len__(self):
    return len(self._extractions)

  def __iter__(self):
    return (Extraction(self, i) for i in xrange(0, len(self)))
  
  def num_extractions(self):
    return len(self)
 
  def num_feats(self):
    return 0 if self.feats is None else self.feats.shape[1]
    
  def _extract(self, sents):
    for sent in sents:
      for ext in self._apply(sent):
        yield ext

  def _apply(self, sent):
    raise NotImplementedError()
            
  def _get_features(self):
      raise NotImplementedError()    
    
  def extract_features(self, *args):
    f_index = self._get_features(args)
    # Apply the feature generator, constructing a sparse matrix incrementally
    # Note that lil_matrix should be relatively efficient as we proceed row-wise
    self.feats = sparse.lil_matrix((self.num_extractions(), len(f_index)))    
    for j,feat in enumerate(f_index.keys()):
      self.feat_index[j] = feat
      for i in f_index[feat]:
        self.feats[i,j] = 1
    return self.feats
      
  def generate_mindtagger_items(self, samp, probs):
    raise NotImplementedError()
    
  def mindtagger_format(self):
    raise NotImplementedError()

  def dump_extractions(self, loc):
    if os.path.isfile(loc):
      warnings.warn("Overwriting file {}".format(loc))
    with open(loc, 'w+') as f:
      cPickle.dump(self._extractions, f)
    
  def __repr__(self):
    return '\n'.join(str(e) for e in self._extractions)

###################################################################
############################ RELATIONS ############################
###################################################################

# Alias for relation
Relation = Extraction

class relation_internal(extraction_internal):
  def __init__(self, e1_idxs, e2_idxs, e1_label, e2_label, sent, xt):
    self.e1_idxs = e1_idxs
    self.e2_idxs = e2_idxs
    self.e1_label = e1_label
    self.e2_label = e2_label
    super(relation_internal, self).__init__([self.e1_idxs, self.e2_idxs],
                                            [self.e1_label, self.e2_label], 
                                             sent, xt)

  def __repr__(self):
    return '<Relation: {}{} - {}{}>'.format([self.words[i] for i in self.e1_idxs], 
      self.e1_idxs, [self.words[i] for i in self.e2_idxs], self.e2_idxs)

class Relations(Extractions):
  def __init__(self, content, matcher1=None, matcher2=None):
    if matcher1 is not None and matcher2 is not None:
      if not issubclass(matcher1.__class__, Matcher):
        warnings.warn("matcher1 is not a Matcher subclass")
      if not issubclass(matcher2.__class__, Matcher):
        warnings.warn("matcher2 is not a Matcher subclass")
      self.e1 = matcher1
      self.e2 = matcher2
    super(Relations, self).__init__(content)
  
  def __getitem__(self, i):
    return Relation(self, i)  
  
  def _apply(self, sent):
    xt = corenlp_to_xmltree(sent)
    for e1_idxs, e1_label in self.e1.apply(sent):
      for e2_idxs, e2_label in self.e2.apply(sent):
        yield relation_internal(e1_idxs, e2_idxs, e1_label, e2_label, sent, xt)
  
  def _get_features(self, method='treedlib'):
    get_feats = compile_relation_feature_generator()
    f_index = defaultdict(list)
    for j,ext in enumerate(self._extractions):
      for feat in get_feats(ext.root, ext.e1_idxs, ext.e2_idxs):
        f_index[feat].append(j)
    return f_index
    
  def generate_mindtagger_items(self, samp, probs):
    for i, p in zip(samp, probs):
      item = self[i]      
      yield dict(
        ext_id          = item.id,
        doc_id          = item.doc_id,
        sent_id         = item.sent_id,
        words           = json.dumps(item.words),
        e1_idxs         = json.dumps(item.e1_idxs),
        e1_label        = item.e1_label,
        e2_idxs         = json.dumps(item.e2_idxs),
        e2_label        = item.e2_label,
        probability     = p
      )
      
  def mindtagger_format(self):
    s1 = """
         <mindtagger-highlight-words index-array="item.e1_idxs" array-format="json" with-style="background-color: yellow;"/>
         <mindtagger-highlight-words index-array="item.e2_idxs" array-format="json" with-style="background-color: cyan;"/>
         """
    s2 = """
         <strong>{{item.e1_label}} -- {{item.e2_label}}</strong>
         """
    return {'style_block' : s1, 'title_block' : s2}
    
##################################################################
############################ ENTITIES ############################
##################################################################     

# Alias for Entity
Entity = Extraction

class entity_internal(extraction_internal):
  def __init__(self, idxs, label, sent, xt):
    self.idxs = idxs
    self.label = label
    super(entity_internal, self).__init__([idxs], [label], sent, xt)

  def __repr__(self):
    return '<Entity: {}{}>'.format([self.words[i] for i in self.idxs], self.idxs)


class Entities(Extractions):
  def __init__(self, content, matcher=None):
    if matcher is not None:
      if not issubclass(matcher.__class__, Matcher):
        warnings.warn("e is not a Matcher subclass")
      self.e = matcher
    super(Entities, self).__init__(content)
  
  def __getitem__(self, i):
    return Entity(self, i)  
  
  def _apply(self, sent):
    xt = corenlp_to_xmltree(sent)
    for e_idxs, e_label in self.e.apply(sent):
      yield entity_internal(e_idxs, e_label, sent, xt)        
  
  def _get_features(self, method='treedlib'):
    get_feats = compile_entity_feature_generator()
    f_index = defaultdict(list)
    for j,ext in enumerate(self._extractions):
      for feat in get_feats(ext.root, ext.idxs):
        f_index[feat].append(j)
      for feat in get_ddlib_feats(ext, ext.idxs):
        f_index["DDLIB_" + feat].append(j)
    return f_index    
  
  def generate_mindtagger_items(self, samp, probs):
    for i, p in zip(samp, probs):
      item = self[i]    
      yield dict(
        ext_id          = item.id,
        doc_id          = item.doc_id,
        sent_id         = item.sent_id,
        words           = json.dumps(item.words),
        idxs            = json.dumps(item.idxs),
        label           = item.label,
        probability     = p
      )
      
  def mindtagger_format(self):
    s1 = """
         <mindtagger-highlight-words index-array="item.idxs" array-format="json" with-style="background-color: cyan;"/>
         """
    s2 = """
         <strong>{{item.label}} candidate</strong>
         """
    return {'style_block' : s1, 'title_block' : s2}
    

####################################################################
############################ INFERENCE #############################
#################################################################### 

class ExtractionInference:
  def __init__(self, extractions, feats=None):
    self.E = extractions
    if type(feats) == np.ndarray or sparse.issparse(feats):
      self.feats = feats
    elif feats is None:
      try:
        self.feats = self.E.extract_features()
      except:
        raise ValueError("Could not automatically extract features")
    else:
      raise ValueError("Features must be numpy ndarray or sparse")
    self.logger = None
    self.rules = None
    self.X = None
    self.w = None
    self.holdout = []
    self.mindtagger_instance = None

  def num_extractions(self):
    return len(self.E)
  
  def num_rules(self, result='all'):
    if self.rules is None:
      return 0
    vals = np.array(np.sum(np.array(self.rules.data)))
    if result.lower().startswith('pos'):
      return np.sum(vals == 1.)
    if result.lower().startswith('neg'):
      return np.sum(vals == -1.)
    if result.lower().startswith('abs'):
      return np.product(self.rules.shape) - self.rules.nnz
    return self.rules.shape[1]
 
  def num_feats(self):
    return 0 if self.feats is None else self.feats.shape[1]

  def apply_rules(self, rules_f, clear=False):
    """ Apply rule functions given in list
    Allows adding to existing rules or clearing rules with CLEAR=True
    """
    nr_old = self.num_rules() if not clear else 0
    add = sparse.lil_matrix((self.num_extractions(), len(rules_f)))
    self.rules = add if (self.rules is None or clear)\
                     else sparse.hstack([self.rules,add], format = 'lil')
    for i,ext in enumerate(self.E._extractions):    
      for ja,rule in enumerate(rules_f):
        self.rules[i,ja + nr_old] = rule(ext)
        
  def learn_weights(self, nSteps=1000, sample=False, nSamples=100, mu=1e-9, 
                    holdout=0.1, use_sparse = True, verbose=False):
    """
    Uses the R x N matrix of rules and the F x N matrix of features defined
    for the Relations object
    Stacks them, giving the rules a +1 prior (i.e. init value)
    Then runs learning, saving the learned weights
    Holds out a set of variables for testing, either a random fraction or a specific set of indices
    """
    N, R, F = self.num_extractions(), self.num_rules(), self.num_feats()
    if hasattr(holdout, "__iter__"):
        self.holdout = holdout
    elif not hasattr(holdout, "__iter__") and (0 <= holdout < 1):
        self.holdout = np.random.choice(N, np.floor(holdout * N), replace=False)
    else:
        raise ValueError("Holdout must be an array of indices or fraction")
    self.X = sparse.hstack([self.rules, self.feats], format='csr')
    if not use_sparse:
      self.X = np.asarray(self.X.todense())
    w0 = np.concatenate([np.ones(R), np.zeros(F)])
    self.w = learn_ridge_logreg(self.X[np.setdiff1d(range(N), self.holdout),:],
                                nSteps=nSteps, w0=w0, sample=sample,
                                nSamples=nSamples, mu=mu, verbose=verbose)

  def get_link(self, subset=None):
    """
    Get the array of predicted link function values (continuous) given learned weight param w
    Return either all variables or only holdout set
    """
    if self.X is None or self.w is None:
      raise ValueError("Inference has not been run yet")
    if subset is None:
      return self.X.dot(self.w)
    if subset is 'holdout':
      return self.X[self.holdout, :].dot(self.w)
    try:
      return self.X[subset, :].dot(self.w)
    except:
      raise ValueError("subset must be either 'holdout' or an array of indices 0 <= i < {}".format(self.num_extractions()))

  def get_predicted_probability(self, subset=None):
    """
    Get the array of predicted probabilities (continuous) for variables given learned weight param w
    Return either all variables or only holdout set
    """
    return odds_to_prob(self.get_link(subset))
 
  def get_predicted(self, subset=None):
    """
    Get the array of predicted (boolean) variables given learned weight param w
    Return either all variables or only holdout set
    """
    return np.sign(self.get_link(subset))
    
  def _handle_ground_truth(self, ground_truth, holdout_only):
    gt = None
    N = self.num_extractions()
    if len(ground_truth) == N:
      gt = ground_truth[self.holdout] if holdout_only else ground_truth
    elif holdout_only and len(ground_truth) == len(self.holdout):
      gt = ground_truth
    else:
      raise ValueError("{} ground truth labels for {} relations and holdout size {}.".
        format(len(ground_truth), N, len(self.holdout)))
    return gt

  def get_classification_accuracy(self, ground_truth, holdout_only=False):
    """
    Given the labels for the Relations set, return the classification accuracy
    Return either accuracy for all variables or only holdout set
    Note: ground_truth must be an array either the length of the full dataset, or of the holdout
          If the latter, holdout_only must be set to True
    """
    gt = self._handle_ground_truth(ground_truth, holdout_only)
    pred = self.get_predicted('holdout' if holdout_only else None)
    return (np.dot(pred, gt) / len(gt) + 1) / 2

  def get_rule_priority_vote_accuracy(self, ground_truth, holdout_only=False):
    """
    This is to answer the question: 'How well would my rules alone do?'
    I.e. without any features, learning of rule or feature weights, etc.- this serves as a
    natural baseline / quick metric
    Labels are assigned by the first rule that emits one for each relation (based on the order
    of the provided rules list)
    Note: ground_truth must be an array either the length of the full dataset, or of the holdout
          If the latter, holdout_only must be set to True
    """
    R, N = self.num_rules(), self.num_extractions()
    gt = self._handle_ground_truth(ground_truth, holdout_only)
    grid = self.holdout if holdout_only else xrange(N)
    correct = 0
    #TODO: more efficient rule checking for sparse matrix using NONZERO
    dense_rules = self.rules.todense()
    for i in grid:
      for j in xrange(R):
        if dense_rules[i,j] != 0:
          correct += 1 if dense_rules[i,j] == gt[j] else 0
          break
    return float(correct) / len(gt)
    
  def _plot_prediction_probability(self, probs):
    plt.hist(probs, bins=10, normed=False, facecolor='blue')
    plt.xlim((0,1))
    plt.xlabel("Probability")
    plt.ylabel("# Predictions")
    
  def _plot_accuracy(self, probs, ground_truth):
    x = 0.1 * (1 + np.array(range(10)))
    bin_assign = [x[i] for i in np.digitize(probs, x)]
    correct = ((2*(probs >= 0.5) - 1) == ground_truth)
    correct_prob = [np.mean(correct[bin_assign == p]) for p in x]
    plt.plot(x, x, 'b--', x, correct_prob, 'ro-')
    plt.xlabel("Probability")
    plt.ylabel("Accuracy")

  def plot_calibration(self, ground_truth = None):
    """
    Show classification accuracy and probability histogram plots
    Note: ground_truth must be an array either the length of the full dataset, or of the holdout
    """
    has_holdout, has_gt = (len(self.holdout) > 0), (ground_truth is not None)
    n_plots = 1 + has_holdout + (has_holdout and has_gt)
    # Whole set histogram
    plt.subplot(1,n_plots,1)
    self._plot_prediction_probability(self.get_predicted_probability())
    plt.title("(a) # Predictions (whole set)")
    # Hold out histogram
    if has_holdout:
      holdout_probs = self.get_predicted_probability('holdout')
      plt.subplot(1,n_plots,2)
      self._plot_prediction_probability(holdout_probs)
      plt.title("(b) # Predictions (holdout set)")
      if has_gt:
        gt = self._handle_ground_truth(ground_truth, holdout_only = True)
        plt.subplot(1,n_plots,3)
        self._plot_accuracy(self, holdout_probs, gt)
        plt.title("(c) Accuracy (holdout set)")
    plt.show()
    
  def open_mindtagger(self, num_sample = 100, **kwargs):
    self.mindtagger_instance = MindTaggerInstance(self.E.mindtagger_format())
    N = self.num_extractions()
    samp = np.random.choice(N, num_sample, replace=False) if N > num_sample else range(N)
    probs = self.get_predicted_probability(subset=samp)
    return self.mindtagger_instance.open_mindtagger(self.E.generate_mindtagger_items,
                                                    samp, probs, **kwargs)
  
  def get_mindtagger_tags(self):
    return self.mindtagger_instance.get_mindtagger_tags()
    

####################################################################
############################ ALGORITHMS ############################
#################################################################### 

#
# Logistic regression algs
# Ported from Chris's Julia notebook...
#
def log_odds(p):
  """This is the logit function"""
  return np.log(p / (1.0 - p))

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
  N, R = X.shape
  t = np.zeros(N)
  f = np.zeros(N)

  # Take samples of random variables
  idxs = np.round(np.random.rand(nSamples) * (N-1)).astype(int)
  ct = np.bincount(idxs)
  # Estimate probability of correct assignment
  increment = np.random.rand(nSamples) < odds_to_prob(X[idxs, :].dot(w))
  increment_f = -1. * (increment - 1)
  t[idxs] = increment * ct[idxs]
  f[idxs] = increment_f * ct[idxs]
  
  return t, f

def exact_data(X, w):
  """
  We calculate the exact conditional probability of the decision variables in
  logistic regression; see sample_data
  """
  t = odds_to_prob(X.dot(w))
  return t, 1-t

def abs_csr(X):
  """ Element-wise absolute value of csr matrix """
  X_abs = X.copy()
  X_abs.data = np.abs(X_abs.data)
  return X_abs

def transform_sample_stats(Xt, t, f, Xt_abs = None):
  """
  Here we calculate the expected accuracy of each rule/feature
  (corresponding to the rows of X) wrt to the distribution of samples S:

    E_S[ accuracy_i ] = E_(t,f)[ \frac{TP + TN}{TP + FP + TN + FN} ]
                      = \frac{X_{i|x_{ij}>0}*t - X_{i|x_{ij}<0}*f}{t+f}
                      = \frac12\left(\frac{X*(t-f)}{t+f} + 1\right)
  """
  if Xt_abs is None:
    Xt_abs = abs_csr(Xt) if sparse.issparse(Xt) else abs(Xt)
  n_pred = Xt_abs.dot(t+f)
  m = (1. / (n_pred + 1e-8)) * (Xt.dot(t) - Xt.dot(f))
  p_correct = (m + 1) / 2
  return p_correct, n_pred

def learn_ridge_logreg(X, nSteps, w0=None, sample=True, nSamples=100, mu=1e-9, 
                       verbose=False):
  """We perform SGD wrt the weights w"""
  if type(X) != np.ndarray and not sparse.issparse(X):
    raise TypeError("Inputs should be np.ndarray type or scipy sparse.")
  N, R = X.shape

  # We initialize w at 1 for rules & 0 for features
  # As a default though, if no w0 provided, we initialize to all zeros
  w = np.zeros(R) if w0 is None else w0
  g = np.zeros(R)
  l = np.zeros(R)

  # Pre-generate other matrices
  Xt = X.transpose()
  Xt_abs = abs_csr(Xt) if sparse.issparse(Xt) else np.abs(Xt)
  
  # Take SGD steps
  for step in range(nSteps):
    if step % 100 == 0 and verbose:    
      if step % 500 == 0:
        print "\nLearning epoch = ",
      print "%s\t" % step,
      

    # Get the expected rule accuracy
    t,f = sample_data(X, w, nSamples=nSamples) if sample else exact_data(X, w)
    p_correct, n_pred = transform_sample_stats(Xt, t, f, Xt_abs)

    # Get the "empirical log odds"; NB: this assumes one is correct, clamp is for sampling...
    l = np.clip(log_odds(p_correct), -10, 10)

    # SGD step, with \ell_2 regularization, and normalization by the number of samples
    g0 = (n_pred*(w - l)) / np.sum(n_pred) + mu*w

    # Momentum term for faster training
    g = 0.95*g0 + 0.05*g

    # Update weights
    w -= 0.01*g
  return w

def main():
  txt = "Han likes Luke and a wookie. Han Solo don\'t like bounty hunters."
  parser = SentenceParser()
  sents = list(parser.parse(txt))

  g = DictionaryMatch('G', ['Han Solo', 'Luke', 'wookie'])
  b = DictionaryMatch('B', ['Bounty Hunters'])

  print "***** Relation 0 *****"
  R = Relations(g, b, sents)
  print R
  print R[0].tagged_sent
  
  print "***** Relation 1 *****"
  R = Relations(g, g, sents)
  print R
  for r in R:
      print r.tagged_sent
  
  print "***** Entity *****"
  E = Entities(g, sents)
  print E                
  for e in E:
      print e.tagged_sent
      
  print "***** Regex *****"
  pattern = "l[a-zA-z]ke"
  rm = RegexMatch('L', pattern, match_attrib='lemmas', ignore_case=True)
  L = Entities(rm, sents)
  for er in L:
      print er.tagged_sent
      
  print "***** Dict + Regex *****"
  pattern = "VB[a-zA-Z]?"
  vbz = RegexMatch('verbs', pattern, match_attrib='poses', ignore_case=True)
  DR = Entities(MultiMatcher(b,vbz), sents)
  for dr in DR:
      print dr.tagged_sent

if __name__ == '__main__':
  main()
