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
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                             'treedlib'))
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
    return (self[i] for i in xrange(0, len(self)))
  
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

class Relation(Candidate):
  def __init__(self, *args, **kwargs):
    super(Relation, self).__init__(*args, **kwargs) 

  def mention(self, m, attribute='words'):
    if attribute is 'text':
      raise ValueError("Cannot get mention against text")
    if m not in [1, 2]:
      raise ValueError("Mention number must be 1 or 2")
    try:
      seq = self.__getattr__(attribute)
      idxs = self.e1_idxs if m == 1 else self.e2_idxs
      return [seq[i] for i in idxs]
    except:
      raise ValueError("Invalid attribute")
      
  def mention1(self, attribute='words'):
      return self.mention(1, attribute)

  def mention2(self, attribute='words'):
      return self.mention(2, attribute)
  
  def __repr__(self):
      hdr = str(self.C._candidates[self.id])
      return "{0}\nWords: {0}\nLemmas: {0}\nPOSES: {0}".format(hdr, self.words,
                                                               self.lemmas,
                                                               self.poses)

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
      if not issubclass(matcher1.__class__, CandidateExtractor):
        warnings.warn("matcher1 is not a CandidateExtractor subclass")
      if not issubclass(matcher2.__class__, CandidateExtractor):
        warnings.warn("matcher2 is not a CandidateExtractor subclass")
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
        words           = json.dumps(corenlp_cleaner(item.words)),
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

class Entity(Candidate):
  def __init__(self, *args, **kwargs):
    super(Entity, self).__init__(*args, **kwargs) 

  def get_attr_seq(self, attribute, idxs):
    if attribute is 'text':
      raise ValueError("Cannot get indexes against text")
    try:
      seq = self.__getattr__(attribute)
      return [seq[i] for i in idxs]
    except:
      raise ValueError("Invalid attribute or index range")  

  def mention(self, attribute='words'):
    return self.get_attr_seq(attribute, self.idxs)
      
  def pre_window(self, attribute='words', n=3):
    b = np.min(self.idxs)
    s = [b - i for i in range(1, min(b+1,n+1))]
    return self.get_attr_seq(attribute, s)
  
  def post_window(self, attribute='words', n=3):
    b = len(self.words) - np.max(self.idxs)
    s = [np.max(self.idxs) + i for i in range(1, min(b,n+1))]
    return self.get_attr_seq(attribute, s)
  
  def __repr__(self):
      hdr = str(self.C._candidates[self.id])
      return "{0}\nWords: {1}\nLemmas: {2}\nPOSES: {3}".format(hdr, self.words,
                                                               self.lemmas,
                                                               self.poses)    
    
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
      if not issubclass(matcher.__class__, CandidateExtractor):
        warnings.warn("matcher is not a CandidateExtractor subclass")
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
        words           = json.dumps(corenlp_cleaner(item.words)),
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
  def __init__(self, log_id, lf_names, gt_idxs, gt, pred):
    self.id = log_id
    self.lf_names = lf_names
    self.gt_idxs = gt_idxs
    self.set_metrics(gt, pred)
  def set_metrics(self, gt, pred):
    self.precision, self.recall = precision(gt, pred), recall(gt, pred)
    self.f1 = f1_score(prec = self.precision, rec = self.recall)
  def num_lfs(self):
    return len(self.lf_names)
  def num_gt(self):
    return len(self.gt_idxs)
  def table_entry(self):
    html = ["<tr>"]
    html.append("<td>{0}</td>".format(self.id))
    html.append("<td>{0}</td>".format(self.num_lfs()))
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
    html.extend("<tr><td>{0}</td></tr>".format(lf) for lf in self.lf_names)
    html.append("</table>")
    return ''.join(html)

class SideTables:
  def __init__(self, table1, table2):
    self.t1, self.t2 = table1, table2
  def _repr_html_(self):
    t1_html = self.t1._repr_html_()
    t2_html = self.t2._repr_html_()
    t1_html = t1_html[:6] + " style=\"float: left\"" + t1_html[6:] 
    t2_html = t2_html[:6] + " style=\"float: left\"" + t2_html[6:] 
    return t1_html + t2_html
    

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

class DDLiteModel:
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
    self.lf_matrix = None
    self.lf_names = []
    self.X = None
    self._w_fit = None
    self.w = None
    self.holdout = np.array([])
    self.validation = np.array([])
    self.test = np.array([])
    self._current_mindtagger_samples = np.array([])
    self._mindtagger_labels = np.zeros((self.num_candidates()))
    self._tags = []
    self._gold_labels = np.zeros((self.num_candidates()))
    self.mindtagger_instance = None

  def num_candidates(self):
    return len(self.C)
    
  def num_feats(self):
    return self.feats.shape[1]
  
  def num_lfs(self, result='all'):
    if self.lf_matrix is None:
      return 0
    return self.lf_matrix.shape[1]

  def set_gold_labels(self, gold):
    """ Set gold labels for all candidates 
    May abstain with 0, and all other labels are -1 or 1
    """
    N = self.num_candidates()
    gold_f = np.ravel(gold)
    if gold_f.shape != (N,) or not np.all(np.in1d(gold_f, [-1,0,1])):
      raise ValueError("Must have {} gold labels in [-1, 0, 1]".format(N))    
    self._gold_labels = gold_f
    
  def set_mindtagger_labels(self, mt):
    """ Set MT labels for all candidates 
    May abstain with 0, and all other labels are -1 or 1
    """
    N = self.num_candidates()
    mt_f = np.ravel(mt)
    if mt_f.shape != (N,) or not np.all(np.in1d(mt_f, [-1,0,1])):
      raise ValueError("Must have {} MT labels in [-1, 0, 1]".format(N))    
    self._mindtagger_labels = mt_f
    
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
    if subset is 'test':
      gt_all = gt_all[self.test]
      has_gt = (gt_all != 0)
      return self.test[has_gt], gt_all[has_gt]
    if subset is 'validation':
      gt_all = gt_all[self.validation]
      has_gt = (gt_all != 0)
      return self.validation[has_gt], gt_all[has_gt]
    try:
      gt_all = gt_all[subset]
      has_gt = (gt_all != 0)
      return subset[has_gt], gt_all[has_gt]
    except:
      raise ValueError("subset must be 'test', 'validation' or an array of\
                       indices 0 <= i < {}".format(self.num_candidates()))
                       
  def set_lf_matrix(self, lf_matrix, names, clear=False):
    try:
      add = sparse.lil_matrix(lf_matrix)
    except:
      raise ValueError("Could not convert lf_matrix to sparse matrix")
    if add.shape[0] != self.num_candidates():
      raise ValueError("lf_matrix must have one row per candidate")
    if len(names) != add.shape[1]:
      raise ValueError("Must have one name per lf_matrix column")
    if self.lf_matrix is None or clear:
      self.lf_matrix = add
      self.lf_names = names
    else:
      self.lf_matrix = sparse.hstack([self.lf_matrix,add], format = 'lil')
      self.lf_names.extend(names)

  def apply_lfs(self, lfs_f, clear=False):
    """ Apply labeler functions given in list
    Allows adding to existing LFs or clearing LFs with CLEAR=True
    """
    add = sparse.lil_matrix((self.num_candidates(), len(lfs_f)))
    for i,c in enumerate(self.C):    
      for j,lf in enumerate(lfs_f):
        add[i,j] = lf(c)
    add_names = [lab.__name__ for lab in lfs_f]
    self.set_lf_matrix(add, add_names, clear)
    
  def delete_lf(self, lf):
    """ Delete LF by index or name """
    if isinstance(lf, str):
      try:
        lf = self.lf_names.index(lf)
      except:
        raise ValueError("{} is not a valid labeling function name".format(lf))
    if isinstance(lf, int):
      try:
        lf_csc = self.lf_matrix.tocsc()
        other_idx = np.concatenate((range(lf), range(lf+1, self.num_lfs())))
        self.lf_matrix = (lf_csc[:, other_idx]).tolil()
        self.lf_names.pop(lf)
      except:
        raise ValueError("{} is not a valid LF index".format(lf))
    else:
      raise ValueError("lf must be a string name or integer index")
    
  def _cover(self, idxs=None):
    idxs = self.devset() if idxs is None else idxs
    try:
      return [np.ravel((self.lf_matrix[idxs,:] == lab).sum(1))
              for lab in [1,-1]]
    except:
      raise ValueError("Invalid indexes for cover")

  def coverage(self, cov=None, idxs=None):
    cov = self._cover(idxs) if cov is None else cov    
    return float(np.sum((cov[0] + cov[1]) > 0)) / len(self.devset())

  def overlap(self, cov=None, idxs=None):    
    cov = self._cover(idxs) if cov is None else cov    
    return float(np.sum((cov[0] + cov[1]) > 1)) / len(self.devset())

  def conflict(self, cov=None, idxs=None):    
    cov = self._cover(idxs) if cov is None else cov    
    return float(np.sum(np.multiply(cov[0], cov[1]) > 0)) / len(self.devset())

  def print_lf_stats(self, idxs=None):
    """
    Returns basic summary statistics of the LFs as applied to the current set of candidates
    * Coverage = % of candidates that have at least one label
    * Overlap  = % of candidates labeled by > 1 LFs
    * Conflict = % of candidates with conflicting labels
    """
    cov = self._cover(idxs)
    print "LF stats on dev set" if idxs is None else "LF stats on idxs"
    print "Coverage:\t{:.3f}%\nOverlap:\t{:.3f}%\nConflict:\t{:.3f}%".format(
            100. * self.coverage(cov), 
            100. * self.overlap(cov),
            100. * self.conflict(cov))

  def _plot_coverage(self, cov):
    cov_ct = [np.sum(x > 0) for x in cov]
    tot_cov = self.coverage(cov)
    idx, bar_width = np.array([1, -1]), 1
    plt.bar(idx, cov_ct, bar_width, color='b')
    plt.xlim((-1.5, 2.5))
    plt.xlabel("Label type")
    plt.ylabel("# candidates with at least one of label type")
    plt.xticks(idx + bar_width * 0.5, ("Positive", "Negative"))
    return tot_cov * 100.
    
  def _plot_conflict(self, cov):
    x, y = cov
    tot_conf = self.conflict(cov)
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

  def plot_lf_stats(self):
    """ Show plots for evaluating LF quality
    Coverage bar plot, overlap histogram, and conflict heat map
    """
    if self.lf_matrix is None:
      raise ValueError("No LFs applied yet")
    n_plots = 2
    cov = self._cover()
    # LF coverage
    plt.subplot(1,n_plots,1)
    tot_cov = self._plot_coverage(cov)
    plt.title("(a) Label balance (candidate coverage: {:.2f}%)".format(tot_cov))
    # LF conflict
    plt.subplot(1,n_plots,2)
    tot_conf = self._plot_conflict(cov)
    plt.title("(b) Label heat map (candidates with conflict: {:.2f}%)".format(tot_conf))
    # Show plots    
    plt.show()

  def _lf_conf(self, lf_idx):
    lf_csc = self.lf_matrix.tocsc()
    other_idx = np.concatenate((range(lf_idx),range(lf_idx+1, self.num_lfs())))
    agree = lf_csc[:, other_idx].multiply(lf_csc[:, lf_idx])
    return float((np.ravel((agree == -1).sum(1)) > 0).sum()) / self.num_candidates()
    
  def top_conflict_lfs(self, n=10):
    """ Show the LFs with the highest mean conflicts per candidate """
    d = {nm : self._lf_conf(i) for i,nm in enumerate(self.lf_names)}
    tab = DictTable(sorted(d.items(), key=lambda t:t[1], reverse=True))
    tab.set_num(n)
    tab.set_title("Labeling function", "Fraction of candidates where LF has conflict")
    return tab
    
  def _abstain_frac(self, lf_idx):
    ds = self.devset()
    lf_csc = abs_sparse(self.lf_matrix.tocsc()[ds,lf_idx])
    return 1 - float((lf_csc == 1).sum()) / len(ds)
    
  def lowest_coverage_lfs(self, n=10):
    """ Show the LFs with the highest fraction of abstains """
    d = {nm : self._abstain_frac(i) for i,nm in enumerate(self.lf_names)}
    tab = DictTable(sorted(d.items(), key=lambda t:t[1], reverse=True))
    tab.set_num(n)
    tab.set_title("Labeling function", "Fraction of abstained votes")
    return tab

  def _lf_acc(self, lf_idx):
    ds = self.devset()
    gt = self.get_ground_truth('resolve')
    pred = (self.lf_matrix.tocsc()[:,lf_idx].todense())
    has_label = np.where(pred != 0)
    has_gt = np.where(gt != 0)
    # Get labels/gt for candidates in dev set, with label, with gt
    gd_idxs = np.intersect1d(has_label, ds)
    gd_idxs = np.intersect1d(has_gt, gd_idxs)
    gt = np.ravel(gt[gd_idxs])
    pred = np.ravel(pred[gd_idxs])
    n_neg = np.sum(pred == -1)
    n_pos = np.sum(pred == 1)
    neg_acc = 0 if n_neg == 0 else float(np.sum((pred == -1) * (gt == -1))) / n_neg
    pos_acc = 0 if n_pos == 0 else float(np.sum((pred == 1) * (gt == 1))) / n_pos
    return (pos_acc, n_pos, neg_acc, n_neg)
    
  def lowest_empirical_accuracy_lfs(self, n=10):
    """ Show the LFs with the lowest accuracy compared to ground truth """
    d = {nm : self._lf_acc(i) for i,nm in enumerate(self.lf_names)}
    tab_pos = DictTable(sorted(d.items(), key=lambda t:t[1][0]))
    for k in tab_pos:
      tab_pos[k] = "{:.3f} (n={})".format(tab_pos[k][0], tab_pos[k][1])
    tab_pos.set_num(n)
    tab_pos.set_title("Labeling function", "Empirical LF positive-class accuracy")
    
    tab_neg = DictTable(sorted(d.items(), key=lambda t:t[1][2]))
    for k in tab_neg:
      tab_neg[k] = "{:.3f} (n={})".format(tab_neg[k][2], tab_neg[k][3])
    tab_neg.set_num(n)
    tab_neg.set_title("Labeling function", "Empirical LF negative-class accuracy")
    return SideTables(tab_pos, tab_neg)    
    
  def set_holdout(self, idxs=None, p=0.5):
    if not 0 <= p <= 1:
      raise ValueError("Validate/test split proportions must be in [0,1]")
    if idxs is None:
      self.holdout = np.ravel(np.where(self.has_ground_truth()))
    else:
      try:
        self.holdout = np.ravel(np.arange(self.num_candidates())[idxs])
      except:
        raise ValueError("Indexes must be in range [0, num_candidates()) or be\
                          boolean array of length num_candidates()")
    h = self.holdout.copy()
    np.random.shuffle(h)
    self.validation = h[ : np.floor(p * len(h))]
    self.test = h[np.floor(p * len(h)) : ]
    
  def devset(self):
    return np.ravel(np.setdiff1d(range(self.num_candidates()), self.holdout))

  def learn_weights(self, maxIter=1000, tol=1e-6, sample=False, 
                    n_samples=100, mu=None, n_mu=20, mu_min_ratio=1e-6, 
                    alpha=0, opt_1se=True, use_sparse = True, plot=False, 
                    verbose=False, log=True):
    """
    Uses the N x R matrix of LFs and the N x F matrix of features
    Stacks them, giving the LFs a +1 prior (i.e. init value)
    Then runs learning, saving the learned weights
    Holds out preset set of candidates for evaluation
    """
    N, R, F = self.num_candidates(), self.num_lfs(), self.num_feats()
    self.X = sparse.hstack([self.lf_matrix, self.feats], format='csr')
    if not use_sparse:
      self.X = np.asarray(self.X.todense())
    w0 = np.concatenate([np.ones(R), np.zeros(F)])
    # If a single mu is provided, just fit a single model
    if mu is not None and (not hasattr(mu, '__iter__') or len(mu) == 1):
      mu = mu if not hasattr(mu, '__iter__') else mu[0]
      self.w = learn_elasticnet_logreg(self.X[self.devset(),:],
                                       maxIter=maxIter, tol=tol, w0=w0, 
                                       mu_seq=mu, alpha=alpha, sample=sample,
                                       n_samples=n_samples, verbose=verbose)[mu]
    # TODO: handle args between learning functions better
    elif len(self.validation) > 0: 
      result = learn_elasticnet_logreg(self.X[self.devset(),:],
                                       maxIter=maxIter, tol=tol, w0=w0, 
                                       mu_seq=mu, alpha=alpha, sample=sample,
                                       n_samples=n_samples, verbose=verbose)
      self._w_fit = OrderedDict()
      ValidatedFit = namedtuple('ValidatedFit', ['w', 'P', 'R', 'F1'])
      gt = self.get_ground_truth()[self.validation]
      f1_opt, w_opt = 0, None
      for mu in sorted(result.keys()):
        w = result[mu]
        pred = odds_to_prob(self.X[self.validation,
                                   self.num_lfs():].dot(w[self.num_lfs():]))
        prec, rec = precision(gt, pred), recall(gt, pred)
        f1 = f1_score(prec = prec, rec = rec)
        self._w_fit[mu] = ValidatedFit(w, prec, rec, f1)
        if f1 >= f1_opt:
          w_opt, f1_opt = w, f1
      self.w = w_opt
    else:
      raise ValueError("Must have validation set if single mu not provided")
    if log:
      return self.add_to_log()

  def get_link(self, subset=None, use_lfs=False):
    """
    Get the array of predicted link function values (continuous) given weights
    Return either all candidates, a specified subset, or only validation/test set
    """
    start = 0 if use_lfs else self.num_lfs()
    if self.X is None or self.w is None:
      raise ValueError("Inference has not been run yet")
    if subset is None:
      return self.X[:, start:].dot(self.w[start:])
    if subset is 'test':
      return self.X[self.test, start:].dot(self.w[start:])
    if subset is 'validation':
      return self.X[self.validation, start:].dot(self.w[start:])
    try:
      return self.X[subset, start:].dot(self.w[start:])
    except:
      raise ValueError("subset must be 'test', 'validation', or an array of\
                       indices 0 <= i < {}".format(self.num_candidates()))

  def get_predicted_probability(self, subset=None):
    """
    Get array of predicted probabilities (continuous) given weights
    Return either all candidates, a specified subset, or only validation/test set
    """
    return odds_to_prob(self.get_link(subset))
 
  def get_predicted(self, subset=None):
    """
    Get the array of predicted (boolean) variables given weights
    Return either all variables, a specified subset, or only validation/test set
    """
    return np.sign(self.get_link(subset))

  def get_classification_accuracy(self, gt='resolve', subset=None):
    """
    Given the ground truth, return the classification accuracy
    Return either accuracy for all candidates, a subset, or validation/test set
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
    """
    R = self.num_lfs()
    grid, gt = self.get_labeled_ground_truth(gt, subset)
    correct = 0
    #TODO: more efficient LF checking for sparse matrix using NONZERO
    dense_lfs = self.lf_matrix.todense()
    for i in grid:
      for j in xrange(R):
        if dense_lfs[i,j] != 0:
          correct += 1 if dense_lfs[i,j] == gt[j] else 0
          break
    return float(correct) / len(gt)
    
  def _plot_prediction_probability(self, probs):
    plt.hist(probs, bins=20, normed=False, facecolor='blue')
    plt.xlim((0,1.025))
    plt.xlabel("Probability")
    plt.ylabel("# Predictions")
    
  def _plot_accuracy(self, probs, ground_truth):
    x = 0.1 * np.array(range(11))
    bin_assign = [x[i] for i in np.digitize(probs, x)-1]
    correct = ((2*(probs >= 0.5) - 1) == ground_truth)
    correct_prob = np.array([np.mean(correct[bin_assign == p]) for p in x])
    xc = x[np.isfinite(correct_prob)]
    correct_prob = correct_prob[np.isfinite(correct_prob)]
    plt.plot(x, np.abs(x-0.5) + 0.5, 'b--', xc, correct_prob, 'ro-')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel("Probability")
    plt.ylabel("Accuracy")

  def plot_calibration(self):
    """
    Show classification accuracy and probability histogram plots
    """
    idxs, gt = self.get_labeled_ground_truth('resolve', None)
    has_test, has_gt = (len(self.test) > 0), (len(gt) > 0)
    n_plots = 1 + has_test + (has_test and has_gt)
    # Whole set histogram
    plt.subplot(1,n_plots,1)
    probs = self.get_predicted_probability()
    self._plot_prediction_probability(probs)
    plt.title("(a) # Predictions (whole set)")
    # Hold out histogram
    if has_test:
      plt.subplot(1,n_plots,2)
      self._plot_prediction_probability(probs[self.test])
      plt.title("(b) # Predictions (test set)")
      # Classification bucket accuracy
      if has_gt:
        plt.subplot(1,n_plots,3)
        self._plot_accuracy(probs[idxs], gt)
        plt.title("(c) Accuracy (test set)")
    plt.show()
    
  def _get_all_abstained(self, dev=True):
      idxs = self.devset() if dev else range(self.num_candidates())
      return np.ravel(np.where(np.ravel((self.lf_matrix[idxs,:]).sum(1)) == 0))
    
  def open_mindtagger(self, num_sample=None, abstain=False, **kwargs):
    self.mindtagger_instance = MindTaggerInstance(self.C.mindtagger_format())
    if isinstance(num_sample, int) and num_sample > 0:
      pool = self._get_all_abstained(dev=True) if abstain else self.devset()
      self._current_mindtagger_samples = np.random.choice(pool, num_sample, replace=False)\
                                          if len(pool) > num_sample else pool
    elif not num_sample and len(self._current_mindtagger_samples) < 0:
      raise ValueError("No current MindTagger sample. Set num_sample")
    elif num_sample:
      raise ValueError("Number of samples is integer or None")
    try:
      probs = self.get_predicted_probability(subset=self._current_mindtagger_samples)
    except:
      probs = [None for _ in xrange(len(self._current_mindtagger_samples))]
    tags_l = self.get_ground_truth('resolve')[self._current_mindtagger_samples]
    tags = np.zeros_like(self._mindtagger_labels)
    tags[self._current_mindtagger_samples] = tags_l
    return self.mindtagger_instance.open_mindtagger(self.C.generate_mindtagger_items,
                                                    self._current_mindtagger_samples,
                                                    probs, tags, **kwargs)
  
  def add_mindtagger_tags(self):
    tags = self.mindtagger_instance.get_mindtagger_tags()
    self._tags = tags
    is_tagged = [i for i,tag in enumerate(tags) if 'is_correct' in tag]
    tb = [tags[i]['is_correct'] for i in is_tagged]
    tb = [1 if t else -1 for t in tb]
    self._mindtagger_labels[self._current_mindtagger_samples[is_tagged]] = tb
    
  def add_to_log(self, log_id=None, gt='resolve', subset='test', verb=True):
    if log_id is None:
      log_id = len(self.logger)
    gt_idxs, gt = self.get_labeled_ground_truth(gt, subset)
    pred = self.get_predicted(gt_idxs)    
    self.logger.log(ModelLog(log_id, self.lf_names, gt_idxs, gt, pred))
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
    
CandidateModel = DDLiteModel

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

def sample_data(X, w, n_samples):
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
  idxs = np.round(np.random.rand(n_samples) * (N-1)).astype(int)
  ct = np.bincount(idxs)
  # Estimate probability of correct assignment
  increment = np.random.rand(n_samples) < odds_to_prob(X[idxs, :].dot(w))
  increment_f = -1. * (increment - 1)
  t[idxs] = increment * ct[idxs]
  f[idxs] = increment_f * ct[idxs]
  
  return t, f


def exact_data(X, w, evidence=None):
  """
  We calculate the exact conditional probability of the decision variables in
  logistic regression; see sample_data
  """
  t = odds_to_prob(X.dot(w))
  if evidence is not None:
    t[evidence > 0.0] = 1.0
    t[evidence < 0.0] = 0.0
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

def transform_sample_stats(Xt, t, f, Xt_abs=None):
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

def learn_elasticnet_logreg(X, maxIter=500, tol=1e-6, w0=None, sample=True,
                            n_samples=100, alpha=0, mu_seq=None, n_mu=20,
                            mu_min_ratio=1e-6, rate=0.01, evidence=None,
                            verbose=False):
  """ Perform SGD wrt the weights w
       * w0 is the initial guess for w
       * sample and n_samples determine SGD batch size
       * alpha is the elastic net penalty mixing parameter (0=ridge, 1=lasso)
       * mu is the sequence of elastic net penalties to search over
  """
  if type(X) != np.ndarray and not sparse.issparse(X):
    raise TypeError("Inputs should be np.ndarray type or scipy sparse.")
  N, R = X.shape

  # Pre-generate other matrices
  Xt = X.transpose()
  Xt_abs = abs_sparse(Xt) if sparse.issparse(Xt) else np.abs(Xt)
  
  # Initialize weights if no initial provided
  w0 = np.zeros(R) if w0 is None else w0   
  
  # Check mixing parameter
  if not (0 <= alpha <= 1):
    raise ValueError("Mixing parameter must be in [0,1]")
  
  # Determine penalty parameters  
  if mu_seq is not None:
    mu_seq = np.ravel(mu_seq)
    if not np.all(mu_seq >= 0):
      raise ValueError("Penalty parameters must be non-negative")
    mu_seq.sort()
  else:
    mu_seq = get_mu_seq(n_mu, rate, alpha, mu_min_ratio)

  if evidence is not None:
    evidence = np.ravel(evidence)

  weights = dict()
  # Search over penalty parameter values
  for mu in mu_seq:
    w = w0.copy()
    g = np.zeros(R)
    l = np.zeros(R)
    # Take SGD steps
    for step in range(maxIter):
      if step % 100 == 0 and verbose:    
        if step % 500 == 0:
          print "Learning epoch = ",
        print "%s\t" % step,
        if (step+100) % 500 == 0:
          print "\n",
      
      # Get the expected LF accuracy
      t,f = sample_data(X, w, n_samples=n_samples) if sample else exact_data(X, w, evidence)
      p_correct, n_pred = transform_sample_stats(Xt, t, f, Xt_abs)

      # Get the "empirical log odds"; NB: this assumes one is correct, clamp is for sampling...
      l = np.clip(log_odds(p_correct), -10, 10)

      # SGD step with normalization by the number of samples
      g0 = (n_pred*(w - l)) / np.sum(n_pred)

      # Momentum term for faster training
      g = 0.95*g0 + 0.05*g

      # Check for convergence
      wn = np.linalg.norm(w, ord=2)
      if wn < 1e-12 or np.linalg.norm(g, ord=2) / wn < tol:
        if verbose:
          print "SGD converged for mu={:.3f} after {} steps".format(mu, step)
        break

      # Update weights
      w -= rate*g
      
      # Apply elastic net penalty
      soft = np.abs(w) - alpha * mu
      #          \ell_1 penalty by soft thresholding        |  \ell_2 penalty
      w = (np.sign(w)*np.select([soft>0], [soft], default=0)) / (1+(1-alpha)*mu)
    
    # SGD did not converge    
    else:
      warnings.warn("SGD did not converge for mu={:.3f}. Try increasing maxIter.".format(mu))

    # Store result and set warm start for next penalty
    weights[mu] = w.copy()
    w0 = w
    
  return weights
  
def get_mu_seq(n, rate, alpha, min_ratio):
  mv = (max(float(1 + rate * 10), float(rate * 11)) / (alpha + 1e-3))
  return np.logspace(np.log10(mv * min_ratio), np.log10(mv), n)
  
def cv_elasticnet_logreg(X, nfolds=5, w0=None, mu_seq=None, alpha=0, rate=0.01,
                         mu_min_ratio=1e-6, n_mu=20, opt_1se=True, 
                         verbose=True, plot=True, **kwargs):
  N, R = X.shape
  # Initialize weights if no initial provided
  w0 = np.zeros(R) if w0 is None else w0   
  # Check mixing parameter
  if not (0 <= alpha <= 1):
    raise ValueError("Mixing parameter must be in [0,1]")
  # Determine penalty parameters  
  if mu_seq is not None:
    mu_seq = np.ravel(mu_seq)
    if not np.all(mu_seq >= 0):
      raise ValueError("Penalty parameters must be non-negative")
    mu_seq.sort()
  else:
    mu_seq = get_mu_seq(n_mu, rate, alpha, mu_min_ratio)
  # Partition data
  try:
    folds = np.array_split(np.random.choice(N, N, replace=False), nfolds)
    if len(folds[0]) < 10:
      warnings.warn("Folds are smaller than 10 observations")
  except:
    raise ValueError("Number of folds must be a non-negative integer")
  # Get CV results
  cv_results = defaultdict(lambda : defaultdict(list))
  for nf, test in enumerate(folds):
    if verbose:
      print "Running test fold {}".format(nf)
    train = np.setdiff1d(range(N), test)
    w = learn_elasticnet_logreg(X[train, :], w0=w0, mu_seq=mu_seq, alpha=alpha,
                                rate=rate, verbose=False, **kwargs)
    for mu, wm in w.iteritems():
      spread = 2*np.sqrt(np.mean(np.square(odds_to_prob(X[test,:].dot(wm)) - 0.5)))
      cv_results[mu]['p'].append(spread)
      cv_results[mu]['nnz'].append(np.sum(np.abs(wm) > 1e-12))
  # Average spreads
  p = np.array([np.mean(cv_results[mu]['p']) for mu in mu_seq])
  # Find opt index, sd, and 1 sd rule index
  opt_idx = np.argmax(p)
  p_sd = np.array([np.std(cv_results[mu]['p']) for mu in mu_seq])
  t = np.max(p) - p_sd[opt_idx]
  idx_1se = np.max(np.where(p >= t))
  # Average number of non-zero coefs
  nnzs = np.array([np.mean(cv_results[mu]['nnz']) for mu in mu_seq])
  # glmnet plot
  if plot:
    fig, ax1 = plt.subplots()
    # Plot spread
    ax1.set_xscale('log', nonposx='clip')    
    ax1.scatter(mu_seq[opt_idx], p[opt_idx], marker='*', color='purple', s=500,
                zorder=10, label="Maximum spread: mu={}".format(mu_seq[opt_idx]))
    ax1.scatter(mu_seq[idx_1se], p[idx_1se], marker='*', color='royalblue', 
                s=500, zorder=10, label="1se rule: mu={}".format(mu_seq[idx_1se]))
    ax1.errorbar(mu_seq, p, yerr=p_sd, fmt='ro-', label='Spread statistic')
    ax1.set_xlabel('log(penalty)')
    ax1.set_ylabel('Marginal probability spread: ' + r'$2\sqrt{\mathrm{mean}[(p_i - 0.5)^2]}$')
    ax1.set_ylim(-0.04, 1.04)
    for t1 in ax1.get_yticklabels():
      t1.set_color('r')
    # Plot nnz
    ax2 = ax1.twinx()
    ax2.plot(mu_seq, nnzs, '.--', color='gray', label='Sparsity')
    ax2.set_ylabel('Number of non-zero coefficients')
    ax2.set_ylim(-0.01*np.max(nnzs), np.max(nnzs)*1.01)
    for t2 in ax2.get_yticklabels():
      t2.set_color('gray')
    # Shrink plot for legend
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0+box1.height*0.1, box1.width, box1.height*0.9])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0+box2.height*0.1, box2.width, box2.height*0.9])
    plt.title("{}-fold cross validation for elastic net logistic regression with mixing parameter {}".
              format(nfolds, alpha))
    lns1, lbs1 = ax1.get_legend_handles_labels()
    lns2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1+lns2, lbs1+lbs2, loc='upper center', bbox_to_anchor=(0.5,-0.05),
               scatterpoints=1, fontsize=10, markerscale=0.5)
    plt.show()
  # Train a model using the 1se mu
  mu_opt = mu_seq[idx_1se if opt_1se else opt_idx]
  w_opt = learn_elasticnet_logreg(X, w0=w0, alpha=alpha, rate=rate,
                                  mu_seq=mu_seq, **kwargs)
  return w_opt[mu_opt]

def precision(gt, pred):
  pred, gt = np.ravel(pred), np.ravel(gt)
  pred[pred == 0] = 1
  tp = np.sum((pred == 1) * (gt == 1))
  fp = np.sum((pred == 1) * (gt != 1))
  return 0 if tp == 0 else float(tp) / float(tp + fp)

def recall(gt, pred):
  pred, gt = np.ravel(pred), np.ravel(gt)
  pred[pred == 0] = 1
  tp = np.sum((pred == 1) * (gt == 1))
  p = np.sum(gt == 1)
  return 0 if tp == 0 else float(tp) / float(p)

def f1_score(gt=None, pred=None, prec=None, rec=None):
  if prec is None or rec is None:
    if gt is None or pred is None:
      raise ValueError("Need both gt and pred or both prec and rec")
    pred, gt = np.ravel(pred), np.ravel(gt)
    prec = precision(gt, pred) if prec is None else prec
    rec = recall(gt, pred) if rec is None else rec
  return 0 if (prec * rec == 0) else 2 * (prec * rec)/(prec + rec)

def main():
  txt = "Han likes Luke and a good-wookie. Han Solo don\'t like bounty hunters."
  parser = SentenceParser()
  sents = list(parser.parse(txt))

  g = DictionaryMatch(label='G', dictionary=['Han Solo', 'Luke', 'wookie'])
  b = DictionaryMatch(label='B', dictionary=['Bounty Hunters'])

  print "***** Relation 0 *****"
  R = Relations(sents, g, b)
  print R
  print R[0].mention1()
  print R[0].mention2()
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
      print e.mention('poses')
      print e.tagged_sent
  print E[0]
  print E[0].pre_window()
  print E[0].post_window()

if __name__ == '__main__':
  main()
