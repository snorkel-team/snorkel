# Base Python
import cPickle, json, os, sys, warnings
from collections import defaultdict, OrderedDict
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
############################ CANDIDATES #############################
#####################################################################

class Candidate(object):
  """ Proxy providing an interface into the Candidates class """
  def __init__(self, candidates, ex_id, p=None):
    self.C = candidates
    self.id = ex_id
    self._p = p
  def __getattr__(self, name):
    if name.startswith('prob'):
      return self._p
    return getattr(self.C._candidates[self.id], name)
  def __repr__(self):
    s = str(self.C._candidates[self.id])
    return s if self._p is None else (s + " with probability " + str(self._p))

class candidate_internal(object):
  """
  Base class for a candidate
  See entity_internal and relation_internal for examples
  """
  def __init__(self, all_idxs, labels, sent, xt):
    self.all_idxs = all_idxs
    self.labels = labels
    # Absorb XMLTree and Sentence object attributes for access by LFs
    self.xt = xt
    self.root = self.xt.root
    self.__dict__.update(sent.__dict__)

    # Add some additional useful attributes
    self.tagged_sent = ' '.join(tag_seqs(self.words, self.all_idxs, self.labels))

  def render(self):
    self.xt.render_tree(self.all_idxs)
  
  # Pickling instructions
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
    
class Candidates(object):
  """
  Base class for a collection of candidates
  Sub-classes need to yield candidates from sentences (_apply) and 
  generate features (_get_features)
  See Relations and Entities for examples
  """
  def __init__(self, C):
    """
    Set up learning problem and generate candidates
     * C is a flat list of Sentence objects or a path to pickled candidates
    """
    if isinstance(C, basestring):
      try:
        with open(C, 'rb') as f:
          self._candidates = cPickle.load(f)
      except:
        raise ValueError("No pickled candidates at {}".format(C))
    else:
      self._candidates = list(self._extract_candidates(C))
    self.feats = None
    self.feat_index = {}
  
  def __getitem__(self, i):
    return Candidate(self, i)
    
  def __len__(self):
    return len(self._candidates)

  def __iter__(self):
    return (Candidate(self, i) for i in xrange(0, len(self)))
  
  def num_candidates(self):
    return len(self)
 
  def num_feats(self):
    return 0 if self.feats is None else self.feats.shape[1]
    
  def _extract_candidates(self, sents):
    for sent in sents:
      for cand in self._apply(sent):
        yield cand

  def _apply(self, sent):
    raise NotImplementedError()
            
  def _get_features(self):
      raise NotImplementedError()    
    
  def extract_features(self, *args):
    f_index = self._get_features(args)
    # Apply the feature generator, constructing a sparse matrix incrementally
    # Note that lil_matrix should be relatively efficient as we proceed row-wise
    self.feats = sparse.lil_matrix((self.num_candidates(), len(f_index)))    
    for j,feat in enumerate(f_index.keys()):
      self.feat_index[j] = feat
      for i in f_index[feat]:
        self.feats[i,j] = 1
    return self.feats
      
  def generate_mindtagger_items(self, samp, probs):
    raise NotImplementedError()
    
  def mindtagger_format(self):
    raise NotImplementedError()

  def dump_candidates(self, loc):
    if os.path.isfile(loc):
      warnings.warn("Overwriting file {}".format(loc))
    with open(loc, 'w+') as f:
      cPickle.dump(self._candidates, f)
    
  def __repr__(self):
    return '\n'.join(str(c) for c in self._candidates)

###################################################################
############################ RELATIONS ############################
###################################################################

# Alias for relation
Relation = Candidate

class relation_internal(candidate_internal):
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

class Relations(Candidates):
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
    for j,cand in enumerate(self._candidates):
      for feat in get_feats(cand.root, cand.e1_idxs, cand.e2_idxs):
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
Entity = Candidate

class entity_internal(candidate_internal):
  def __init__(self, idxs, label, sent, xt):
    self.idxs = idxs
    self.label = label
    super(entity_internal, self).__init__([idxs], [label], sent, xt)

  def __repr__(self):
    return '<Entity: {}{}>'.format([self.words[i] for i in self.idxs], self.idxs)


class Entities(Candidates):
  def __init__(self, content, matcher=None):
    if matcher is not None:
      if not issubclass(matcher.__class__, Matcher):
        warnings.warn("matcher is not a Matcher subclass")
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
    for j,cand in enumerate(self._candidates):
      for feat in get_feats(cand.root, cand.idxs):
        f_index[feat].append(j)
      for feat in get_ddlib_feats(cand, cand.idxs):
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
############################ LEARNING ##############################
#################################################################### 

class DictTable(OrderedDict):
  def set_title(self, head1, head2):
    self.title = [head1, head2]
  def set_num(self, n):
    self.num = n
  def _repr_html_(self):
    html = ["<table>"]
    if hasattr(self, 'title'):
      html.append("<tr>")
      html.append("<td><b>{0}</b></td>".format(self.title[0]))
      html.append("<td><b>{0}</b></td>".format(self.title[1]))
      html.append("</tr>")
    items = self.items()[:self.num] if hasattr(self, 'num') else self.items()
    for k, v in items:
      html.append("<tr>")
      html.append("<td>{0}</td>".format(k))
      v = "{:.3f}".format(v) if isinstance(v, float) else v
      html.append("<td>{0}</td>".format(v))
      html.append("</tr>")
    html.append("</table>")
    return ''.join(html)

def log_title(heads=["ID", "# LFs", "# ground truth", "Precision", "Recall", "F1"]):
  html = ["<tr>"]
  html.extend("<td><b>{0}</b></td>".format(h) for h in heads)
  html.append("</tr>")
  return ''.join(html)

class ModelLog:
  def __init__(self, log_id, LF_names, gt_idxs, gt, pred):
    self.id = log_id
    self.LF_names = LF_names
    self.gt_idxs = gt_idxs
    self.set_metrics(gt, pred)
  def set_metrics(self, gt, pred):
    tp = np.sum((pred == 1) * (gt == 1))
    fp = np.sum((pred == 1) * (gt == -1))
    fn = np.sum((pred == -1) * (gt == 1))
    self.precision = 0 if tp == 0 else float(tp) / float(tp + fp)
    self.recall = 0 if tp == 0 else float(tp) / float(tp + fn)
    self.f1 = 2 * (self.precision * self.recall)/(self.precision + self.recall)
  def num_LFs(self):
    return len(self.LF_names)
  def num_gt(self):
    return len(self.gt_idxs)
  def table_entry(self):
    html = ["<tr>"]
    html.append("<td>{0}</td>".format(self.id))
    html.append("<td>{0}</td>".format(self.num_LFs()))
    html.append("<td>{0}</td>".format(self.num_gt()))
    html.append("<td>{:.3f}</td>".format(self.precision))
    html.append("<td>{:.3f}</td>".format(self.recall))
    html.append("<td>{:.3f}</td>".format(self.f1))
    html.append("</tr>")
    return ''.join(html)
  def _repr_html_(self):
    html = ["<table>"]
    html.append(log_title())
    html.append(self.table_entry())
    html.append("</table>")
    html.append("<table>")
    html.append(log_title(["LF"]))
    html.extend("<tr><td>{0}</td></tr>".format(lf) for lf in self.LF_names)
    html.append("</table>")
    return ''.join(html)

class ModelLogger:
  def __init__(self):
    self.logs = []
  def __getitem__(self, i):
    return self.logs[i]
  def __len__(self):
    return len(self.logs)
  def __iter__(self):
    return (self.logs[i] for i in xrange(0, len(self)))
  def log(self, ml):
    if issubclass(ml.__class__, ModelLog):
      self.logs.append(ml)
    else:
      raise ValueError("Log must be subclass of ModelLog")
  def _repr_html_(self):
    html = ["<table>"]
    html.append(log_title())
    html.extend(log.table_entry() for log in self.logs)
    html.append("</table>")
    return ''.join(html)

class CandidateModel:
  def __init__(self, candidates, feats=None):
    self.C = candidates
    if type(feats) == np.ndarray or sparse.issparse(feats):
      self.feats = feats
    elif feats is None:
      try:
        self.feats = self.C.extract_features()
      except:
        raise ValueError("Could not automatically extract features")
    else:
      raise ValueError("Features must be numpy ndarray or sparse")
    self.logger = ModelLogger()
    self.LFs = None
    self.LF_names = []
    self.X = None
    self.w = None
    self.holdout = np.array([])
    self._current_mindtagger_samples = np.array([])
    self._mindtagger_labels = np.zeros((self.num_candidates()))
    self._tags = []
    self._gold_labels = np.zeros((self.num_candidates()))
    self.mindtagger_instance = None

  def num_candidates(self):
    return len(self.C)
    
  def num_feats(self):
    return self.feats.shape[1]
  
  def num_LFs(self, result='all'):
    if self.LFs is None:
      return 0
    return self.LFs.shape[1]

  def set_gold_labels(self, gold):
    """ Set gold labels for all candidates 
    May abstain with 0, and all other labels are -1 or 1
    """
    N = self.num_candidates()
    gold_f = np.ravel(gold)
    if gold_f.shape != (N,) or not np.all(np.in1d(gold_f, [-1,0,1])):
      raise ValueError("Must have {} gold labels in [-1, 0, 1]".format(N))    
    self._gold_labels = gold_f
    
  def get_ground_truth(self, gt='resolve'):
    """ Get ground truth from mindtagger, gold, or both with priority gold """
    if gt.lower() == 'resolve':
      return np.array([g if g != 0 else m for g,m in
                       zip(self._gold_labels, self._mindtagger_labels)])
    if gt.lower() == 'mindtagger':
      return self._mindtagger_labels
    if gt.lower() == 'gold':
      return self._gold_labels
    raise ValueError("Unknown ground truth type: {}".format(gt))
 
  def has_ground_truth(self):
    """ Get boolean array of which candidates have some ground truth """
    return self.get_ground_truth() != 0
    
  def get_labeled_ground_truth(self, gt='resolve', subset=None):
    """ Get indices and labels of subset which has ground truth """
    gt_all = self.get_ground_truth(gt)
    if subset is None:
      has_gt = (gt_all != 0)
      return np.ravel(np.where(has_gt)), gt_all[has_gt]
    if subset is 'holdout':
      gt_all = gt_all[self.holdout]
      has_gt = (gt_all != 0)
      return self.holdout[has_gt], gt_all[has_gt]
    try:
      gt_all = gt_all[subset]
      has_gt = (gt_all != 0)
      return subset[has_gt], gt_all[has_gt]
    except:
      raise ValueError("subset must be either 'holdout' or an array of\
                       indices 0 <= i < {}".format(self.num_candidates()))

  def apply_LFs(self, LFs_f, clear=False):
    """ Apply labeler functions given in list
    Allows adding to existing LFs or clearing LFs with CLEAR=True
    """
    nr_old = self.num_LFs() if not clear else 0
    add = sparse.lil_matrix((self.num_candidates(), len(LFs_f)))
    add_names = [lab.__name__ for lab in LFs_f]
    if self.LFs is None or clear:
      self.LFs = add
      self.LF_names = add_names
    else:
      self.LFs = sparse.hstack([self.LFs,add], format = 'lil')
      self.LF_names.extend(add_names)
    for i,c in enumerate(self.C._candidates):    
      for j,LF in enumerate(LFs_f):
        self.LFs[i,j + nr_old] = LF(c)
    
  def _coverage(self):
    return [np.ravel((self.LFs == lab).sum(1)) for lab in [1,-1]]

  def _plot_coverage(self, cov):
    cov_ct = [np.sum(x > 0) for x in cov]
    tot_cov = float(np.sum((cov[0] + cov[1]) > 0)) / self.num_candidates()
    idx, bar_width = np.array([1, -1]), 1
    plt.bar(idx, cov_ct, bar_width, color='b')
    plt.xlim((-1.5, 2.5))
    plt.xlabel("Label type")
    plt.ylabel("# candidates with at least one of label type")
    plt.xticks(idx + bar_width * 0.5, ("Positive", "Negative"))
    return tot_cov * 100.
    
  def _plot_overlap(self):
    tot_ov = float(np.sum(abs_sparse(self.LFs).sum(1) > 1)) / self.num_candidates()
    cts = abs_sparse(self.LFs).sum(1)
    plt.hist(cts, bins=min(15, self.num_LFs()+1), facecolor='blue')
    plt.xlim((0,np.max(cts)+1))
    plt.xlabel("# positive and negative labels")
    plt.ylabel("# candidates")
    return tot_ov * 100.
    
  def _plot_conflict(self, cov):
    x, y = cov
    tot_conf = float(np.dot(x, y)) / self.num_candidates()
    m = np.max([np.max(x), np.max(y)])
    bz = np.linspace(-0.5, m+0.5, num=m+2)
    H, xr, yr = np.histogram2d(x, y, bins=[bz,bz], normed=False)
    plt.imshow(H, interpolation='nearest', origin='low',
               extent=[xr[0], xr[-1], yr[0], yr[-1]])
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label("# candidates")
    plt.xlabel("# negative labels")
    plt.ylabel("# positive labels")
    plt.xticks(range(m+1))
    plt.yticks(range(m+1))
    return tot_conf * 100.

  def plot_LF_stats(self):
    """ Show plots for evaluating LF quality
    Coverage bar plot, overlap histogram, and conflict heat map
    """
    if self.LFs is None:
      raise ValueError("No LFs applied yet")
    n_plots = 3
    cov = self._coverage()
    # LF coverage
    plt.subplot(1,n_plots,1)
    tot_cov = self._plot_coverage(cov)
    plt.title("(a) Label balance (candidate coverage: {:.2f}%)".format(tot_cov))
    # LF overlap
    plt.subplot(1,n_plots,2)
    tot_ov = self._plot_overlap()
    plt.title("(b) Label count histogram (candidates with overlap: {:.2f}%)".format(tot_ov))
    # LF conflict
    plt.subplot(1,n_plots,3)
    tot_conf = self._plot_conflict(cov)
    plt.title("(c) Label heat map (candidates with conflict: {:.2f}%)".format(tot_conf))
    # Show plots    
    plt.show()

  def _LF_conf(self, LF_idx):
    LF_csc = self.LFs.tocsc()
    other_idx = np.setdiff1d(range(self.num_LFs()), LF_idx)
    agree = LF_csc[:, other_idx].multiply(LF_csc[:, LF_idx])
    return float((np.ravel((agree == -1).sum(1)) > 0).sum()) / self.num_candidates()
    
  def top_conflict_LFs(self, n=10):
    """ Show the LFs with the highest mean conflicts per candidate """
    d = {nm : self._LF_conf(i) for i,nm in enumerate(self.LF_names)}
    tab = DictTable(sorted(d.items(), key=lambda t:t[1], reverse=True))
    tab.set_num(n)
    tab.set_title("Labeling function", "Fraction of candidates where LF has conflict")
    return tab
    
  def _abstain_frac(self, LF_idx):
    LF_csc = abs_sparse(self.LFs.tocsc()[:,LF_idx])
    return 1 - float((LF_csc == 1).sum()) / self.num_candidates()
    
  def lowest_coverage_LFs(self, n=10):
    """ Show the LFs with the highest fraction of abstains """
    d = {nm : self._abstain_frac(i) for i,nm in enumerate(self.LF_names)}
    tab = DictTable(sorted(d.items(), key=lambda t:t[1], reverse=True))
    tab.set_num(n)
    tab.set_title("Labeling function", "Fraction of abstained votes")
    return tab
    
  def _LF_acc(self, LF_idx, subset):
    idxs, gt = self.get_labeled_ground_truth('resolve', subset)
    agree = np.ravel(self.LFs.tocsc()[:,LF_idx].todense())[idxs] * gt    
    n_both = np.sum(agree != 0)
    if n_both == 0:
      return (0, 0)
    return (float(np.sum(agree == 1)) / n_both, n_both)

  def lowest_empirical_accuracy_LFs(self, n=10, subset=None):
    """ Show the LFs with the lowest accuracy compared to ground truth """
    d = {nm : self._LF_acc(i,subset) for i,nm in enumerate(self.LF_names)}
    tab = DictTable(sorted(d.items(), key=lambda t:t[1][0]))
    for k in tab:
      tab[k] = "{:.3f} (n={})".format(tab[k][0], tab[k][1])
    tab.set_num(n)
    tab.set_title("Labeling function", "Empirical LF accuracy")
    return tab    
    
  def set_holdout(self, idxs=None):
    if idxs is None:
      self.holdout = np.ravel(np.where(self.has_ground_truth()))
    else:
      try:
        self.holdout = np.ravel(np.arange(self.num_candidates())[idxs])
      except:
        raise ValueError("Indexes must be in range [0, num_candidates()) or be\
                          boolean array of length num_candidates()")

  def learn_weights(self, nSteps=1000, sample=False, nSamples=100, mu=1e-9,
                    use_sparse = True, verbose=False, log=True):
    """
    Uses the N x R matrix of LFs and the N x F matrix of features
    Stacks them, giving the LFs a +1 prior (i.e. init value)
    Then runs learning, saving the learned weights
    Holds out preset set of candidates for evaluation
    """
    N, R, F = self.num_candidates(), self.num_LFs(), self.num_feats()
    self.X = sparse.hstack([self.LFs, self.feats], format='csr')
    if not use_sparse:
      self.X = np.asarray(self.X.todense())
    w0 = np.concatenate([np.ones(R), np.zeros(F)])
    self.w = learn_ridge_logreg(self.X[np.setdiff1d(range(N), self.holdout),:],
                                nSteps=nSteps, w0=w0, sample=sample,
                                nSamples=nSamples, mu=mu, verbose=verbose)
    if log:
      return self.add_to_log()

  def get_link(self, subset=None):
    """
    Get the array of predicted link function values (continuous) given weights
    Return either all candidates, a specified subset, or only holdout set
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
      raise ValueError("subset must be either 'holdout' or an array of\
                       indices 0 <= i < {}".format(self.num_candidates()))

  def get_predicted_probability(self, subset=None):
    """
    Get array of predicted probabilities (continuous) given weights
    Return either all candidates, a specified subset, or only holdout set
    """
    return odds_to_prob(self.get_link(subset))
 
  def get_predicted(self, subset=None):
    """
    Get the array of predicted (boolean) variables given weights
    Return either all variables, a specified subset, or only holdout set
    """
    return np.sign(self.get_link(subset))

  def get_classification_accuracy(self, gt='resolve', subset=None):
    """
    Given the ground truth, return the classification accuracy
    Return either accuracy for all candidates, a subset, or holdout set
    """
    idxs, gt = self.get_labeled_ground_truth(gt, subset)
    pred = self.get_predicted(idxs)
    return np.mean(gt == pred)

  def get_LF_priority_vote_accuracy(self, gt='resolve', subset=None):
    """
    This is to answer the question: 'How well would my LFs alone do?'
    I.e. without any features, learning of LF or feature weights, etc.- this serves as a
    natural baseline / quick metric
    Labels are assigned by the first LF that emits one for each relation (based on the order
    of the provided LF list)
    Note: ground_truth must be an array either the length of the full dataset, or of the holdout
          If the latter, holdout_only must be set to True
    """
    R = self.num_LFs()
    grid, gt = self.get_labeled_ground_truth(gt, subset)
    correct = 0
    #TODO: more efficient LF checking for sparse matrix using NONZERO
    dense_LFs = self.LFs.todense()
    for i in grid:
      for j in xrange(R):
        if dense_LFs[i,j] != 0:
          correct += 1 if dense_LFs[i,j] == gt[j] else 0
          break
    return float(correct) / len(gt)
    
  def _plot_prediction_probability(self, probs):
    plt.hist(probs, bins=10, normed=False, facecolor='blue')
    plt.xlim((0,1))
    plt.xlabel("Probability")
    plt.ylabel("# Predictions")
    
  def _plot_accuracy(self, probs, ground_truth):
    x = 0.1 * np.array(range(11))
    bin_assign = [x[i] for i in np.digitize(probs, x)]
    correct = ((2*(probs >= 0.5) - 1) == ground_truth)
    correct_prob = np.array([np.mean(correct[bin_assign == p]) for p in x])
    xc = x[np.isfinite(correct_prob)]
    correct_prob = correct_prob[np.isfinite(correct_prob)]
    plt.plot(x, x, 'b--', xc, correct_prob, 'ro-')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel("Probability")
    plt.ylabel("Accuracy")

  def plot_calibration(self):
    """
    Show classification accuracy and probability histogram plots
    """
    idxs, gt = self.get_labeled_ground_truth('resolve', None)
    has_holdout, has_gt = (len(self.holdout) > 0), (len(gt) > 0)
    n_plots = 1 + has_holdout + (has_holdout and has_gt)
    # Whole set histogram
    plt.subplot(1,n_plots,1)
    probs = self.get_predicted_probability()
    self._plot_prediction_probability(probs)
    plt.title("(a) # Predictions (whole set)")
    # Hold out histogram
    if has_holdout:
      plt.subplot(1,n_plots,2)
      self._plot_prediction_probability(probs[self.holdout])
      plt.title("(b) # Predictions (holdout set)")
      # Classification bucket accuracy
      if has_gt:
        plt.subplot(1,n_plots,3)
        self._plot_accuracy(probs[idxs], gt)
        plt.title("(c) Accuracy (holdout set)")
    plt.show()
    
  def open_mindtagger(self, num_sample = None, **kwargs):
    self.mindtagger_instance = MindTaggerInstance(self.C.mindtagger_format())
    if isinstance(num_sample, int):
      N = self.num_candidates()
      self._current_mindtagger_samples = np.random.choice(N, num_sample, replace=False)\
                                          if N > num_sample else range(N)
    elif num_sample is not None:
      raise ValueError("Number of samples is integer or None")
    try:
      probs = self.get_predicted_probability(subset=self._current_mindtagger_samples)
    except:
      probs = [None for _ in xrange(len(self._current_mindtagger_samples))]
    return self.mindtagger_instance.open_mindtagger(self.C.generate_mindtagger_items,
                                                    self._current_mindtagger_samples,
                                                    probs, **kwargs)
  
  def add_mindtagger_tags(self, tags=None):
    tags = self.mindtagger_instance.get_mindtagger_tags()
    self._tags = tags
    is_tagged = [i for i,tag in enumerate(tags) if 'is_correct' in tag]
    tb = [tags[i]['is_correct'] for i in is_tagged]
    tb = [1 if t else -1 for t in tb]
    self._mindtagger_labels[self._current_mindtagger_samples[is_tagged]] = tb
    
  def add_to_log(self, log_id=None, gt='resolve', subset='holdout', verb=True):
    if log_id is None:
      log_id = len(self.logger)
    gt_idxs, gt = self.get_labeled_ground_truth(gt, subset)
    pred = self.get_predicted(gt_idxs)    
    self.logger.log(ModelLog(log_id, self.LF_names, gt_idxs, gt, pred))
    if verb:
      return self.logger[-1]
      
  def show_log(self, idx=None):
    if idx is None:
      return self.logger
    elif isinstance(idx, int):
      try:
        return self.logger[idx]
      except:
        raise ValueError("Index must be for one of {} logs".format(len(self.logger)))
    else:
      raise ValueError("Index must be an integer index or None")
    

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

def abs_sparse(X):
  """ Element-wise absolute value of sparse matrix """
  X_abs = X.copy()
  if sparse.isspmatrix_csr(X) or sparse.isspmatrix_csc(X):
    X_abs.data = np.abs(X_abs.data)
  elif sparse.isspmatrix_lil(X):
    X_abs.data = np.array([np.abs(L) for L in X_abs.data])
  else:
    raise ValueError("Only supports CSR/CSC and LIL matrices")
  return X_abs

def transform_sample_stats(Xt, t, f, Xt_abs = None):
  """
  Here we calculate the expected accuracy of each LF/feature
  (corresponding to the rows of X) wrt to the distribution of samples S:

    E_S[ accuracy_i ] = E_(t,f)[ \frac{TP + TN}{TP + FP + TN + FN} ]
                      = \frac{X_{i|x_{ij}>0}*t - X_{i|x_{ij}<0}*f}{t+f}
                      = \frac12\left(\frac{X*(t-f)}{t+f} + 1\right)
  """
  if Xt_abs is None:
    Xt_abs = abs_sparse(Xt) if sparse.issparse(Xt) else abs(Xt)
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

  # We initialize w at 1 for LFs & 0 for features
  # As a default though, if no w0 provided, we initialize to all zeros
  w = np.zeros(R) if w0 is None else w0
  g = np.zeros(R)
  l = np.zeros(R)

  # Pre-generate other matrices
  Xt = X.transpose()
  Xt_abs = abs_sparse(Xt) if sparse.issparse(Xt) else np.abs(Xt)
  
  # Take SGD steps
  for step in range(nSteps):
    if step % 100 == 0 and verbose:    
      if step % 500 == 0:
        print "\nLearning epoch = ",
      print "%s\t" % step,
      

    # Get the expected LF accuracy
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
  txt = "Han likes Luke and a good-wookie. Han Solo don\'t like bounty hunters."
  parser = SentenceParser()
  sents = list(parser.parse(txt))

  g = DictionaryMatch('G', ['Han Solo', 'Luke', 'wookie'])
  b = DictionaryMatch('B', ['Bounty Hunters'])

  print "***** Relation 0 *****"
  R = Relations(sents, g, b)
  print R
  print R[0].tagged_sent
  
  print "***** Relation 1 *****"
  R = Relations(sents, g, g)
  print R
  for r in R:
      print r.tagged_sent
  
  print "***** Entity *****"
  E = Entities(sents, g)
  print E                
  for e in E:
      print e.tagged_sent
      
  print "***** Regex *****"
  pattern = "l[a-zA-z]ke"
  rm = RegexMatch('L', pattern, match_attrib='lemmas', ignore_case=True)
  L = Entities(sents, rm)
  for er in L:
      print er.tagged_sent
      
  print "***** Dict + Regex *****"
  pattern = "VB[a-zA-Z]?"
  vbz = RegexMatch('verbs', pattern, match_attrib='poses', ignore_case=True)
  n = RegexMatch('neg', "don\'t", match_attrib='text')
  DR = Entities(sents, MultiMatcher(b,vbz,n))
  for dr in DR:
      print dr.tagged_sent

if __name__ == '__main__':
  main()
