# Base Python
import cPickle, json, os, sys, warnings
from collections import defaultdict, OrderedDict, namedtuple
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

# ddlite learning
from ddlite_learning import learn_elasticnet_logreg, odds_to_prob

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
    
  def get_attr_seq(self, attribute, idxs):
    if attribute is 'text':
      raise ValueError("Cannot get indexes against text")
    try:
      seq = self.__getattr__(attribute)
      return [seq[i] for i in idxs]
    except:
      raise ValueError("Invalid attribute or index range")  
    
  def __repr__(self):
    s = str(self.C._candidates[self.id])
    return s if self._p is None else (s + " with probability " + str(self._p))

class candidate_internal(object):
  """
  Base class for a candidate
  See entity_internal and relation_internal for examples
  """
  def __init__(self, all_idxs, labels, sent, xt):
    self.uid = "{}::{}::{}::{}".format(sent.doc_id, sent.sent_id,
                                       all_idxs, labels)
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
    if m not in [1, 2]:
      raise ValueError("Mention number must be 1 or 2")
    return self.get_attr_seq(attribute, self.e1_idxs if m==1 else self.e2_idxs)
    
  def mention1(self, attribute='words'):
      return self.mention(1, attribute)

  def mention2(self, attribute='words'):
      return self.mention(2, attribute)    
    
  def pre_window(self, m, attribute='words', n=3):
    if m not in [1, 2]:
      raise ValueError("Mention number must be 1 or 2")
    idxs = self.e1_idxs if m==1 else self.e2_idxs
    b = np.min(idxs)
    s = [b - i for i in range(1, min(b+1,n+1))]
    return self.get_attr_seq(attribute, s)
  
  def post_window(self, m, attribute='words', n=3):
    if m not in [1, 2]:
      raise ValueError("Mention number must be 1 or 2")
    idxs = self.e1_idxs if m==1 else self.e2_idxs
    b = len(self.words) - np.max(idxs)
    s = [np.max(idxs) + i for i in range(1, min(b,n+1))]
    return self.get_attr_seq(attribute, s)
    
  def pre_window1(self, attribute='words', n=3):
    return self.pre_window(1, attribute, n)

  def pre_window2(self, attribute='words', n=3):
    return self.pre_window(2, attribute, n)

  def post_window1(self, attribute='words', n=3):
    return self.post_window(1, attribute, n)
    
  def post_window2(self, attribute='words', n=3):
    return self.post_window(2, attribute, n)
  
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

class DictTable(OrderedDict):
  def set_title(self, heads):
    self.title = heads
  def set_rows(self, n):
    self.rows = n
  def set_cols(self, n):
    self.cols = n
  def _repr_html_(self):
    html = ["<table>"]
    if hasattr(self, 'title'):
      html.append("<tr>")
      html.extend("<td><b>{0}</b></td>".format(t) for t in self.title)
      html.append("</tr>")
    items = self.items()[:self.rows] if hasattr(self, 'rows') else self.items()
    for k, v in items:
      html.append("<tr>")
      html.append("<td>{0}</td>".format(k))
      html.extend("<td>{0}</td>".format(i) for i in v)
      html.append("</tr>")
    html.append("</table>")
    return ''.join(html)

class SideTables:
  def __init__(self, table1, table2):
    self.t1, self.t2 = table1, table2
  def _repr_html_(self):
    t1_html = self.t1._repr_html_()
    t2_html = self.t2._repr_html_()
    t1_html = t1_html[:6] + " style=\"margin-right: 1%;float: left\"" + t1_html[6:] 
    t2_html = t2_html[:6] + " style=\"float: left\"" + t2_html[6:] 
    return t1_html + t2_html

def log_title(heads=["ID", "# LFs", "Test set size", "Use LFs", "Model", 
                     "Precision", "Recall", "F1"]):
  html = ["<tr>"]
  html.extend("<td><b>{0}</b></td>".format(h) for h in heads)
  html.append("</tr>")
  return ''.join(html)

class ModelLog:
  def __init__(self, log_id, lf_names, use_lfs, model, gt_idxs, gt, pred):
    self.id = log_id
    self.lf_names = lf_names
    self.use_lfs = use_lfs
    self.model = model
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
    html.append("<td>{0}</td>".format(self.use_lfs))
    html.append("<td>{0}</td>".format(self.model))    
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

ValidatedFit = namedtuple('ValidatedFit', ['w', 'P', 'R', 'F1'])

class CandidateGT:
  def __init__(self, candidates, gt_dict=None):
    gt_dict = gt_dict if isinstance(gt_dict, dict) else dict()
    self._gt_dict = OrderedDict()
    self._gt_vec = np.zeros((len(candidates)), dtype=int)
    for i,c in enumerate(candidates):
      l = gt_dict[c.uid] if c.uid in gt_dict else 0
      self._gt_dict[c.uid] = l
      self._gt_vec[i] = l

    self.validation = np.array([], dtype=int)
    self.test = np.array([], dtype=int)
    
    self.training = np.array(range(self.n()), dtype=int)
    
    self.dev_split = 0.7
    self.dev1 = np.array([], dtype=int)
    self.dev2 = np.array([], dtype=int)
    
    self.min_dev = 20
    self.min_val = 20
    self.min_test = 20
      
  def n(self):
    return len(self._gt_vec)
    
  def get_gt_dict(self):
    return self._gt_dict

  def dev_size_warn(self):    
    if len(self.dev1) < self.min_dev or len(self.dev2) < self.min_dev:
      warnings.warn("Dev sets are too small for reliable estimates")
     
  def val_test_size_warn(self):    
    if len(self.validation) < self.min_val or len(self.test) < self.min_test:
      warnings.warn("Validation/test sets are too small for reliable estimates")
      
    
  def _update_training(self):
    self.training = np.setdiff1d(np.array(range(self.n()), dtype=int), 
                                 self.holdout())
    
  def holdout(self):
    return np.concatenate([self.validation, self.test])
    
  def dev(self):
    return np.concatenate([self.dev1, self.dev2])

  def set_holdout(self, idxs=None, validation_frac=0.5):
    """ Set validation and test sets """
    if not 0 <= validation_frac <= 1:
      raise ValueError("Validate/test split proportions must be in [0,1]")
    if idxs is None:
      h = np.ravel(np.where(self._gt_vec != 0))
    else:
      try:
        h = np.ravel(np.arange(self.n())[idxs])
      except:
        raise ValueError("Indexes must be in range [0, num_candidates()) or be\
                          boolean array of length num_candidates()")
    np.random.shuffle(h)
    self.validation = h[ : np.floor(validation_frac * len(h))]
    self.test = h[np.floor(validation_frac * len(h)) : ]
    self._update_training()
    self._update_devs(self.dev_split)
    
  def _update_devs(self, dev_split):
    idxs,_ = self.get_labeled_ground_truth('training')
    np.random.shuffle(idxs)
    self.dev1 = idxs[ : np.floor(dev_split * len(idxs))]
    self.dev2 = idxs[np.floor(dev_split * len(idxs)) : ]
    
  def update_gt(self, gt, idxs=None, uids=None):
    """ Set ground truth for idxs XOR uids to gt. Updates dev sets. """
    # Check input  
    try:
      gt = np.ravel(gt)
    except:
      raise ValueError("gt must be array-like")
    if not np.all(np.in1d(gt, [-1,0,1])):
      raise ValueError("gt must be -1, 0, or 1")
    # Assign gt by indexes  
    if idxs is not None and uids is None:
      if len(idxs) != len(gt):
        raise ValueError("idxs and gt must be same length")
      try:
        self._gt_vec[idxs] = gt
      except:
        raise ValueError("Could not assign gt to idxs")
      k = self._gt_dict.keys()
      for i,label in zip(idxs,gt):
        self._gt_dict[k[i]] = label
    # Assign gt by uid    
    elif uids is not None and idxs is None:
      if len(uids) != len(gt):
        raise ValueError("uids and gt must be same length")
      for uid,label in zip(uids,gt):
        if uid not in self._gt_dict:
          raise ValueError("uid {} not in candidates".format(uid))
        self._gt_dict[uid] = label
      # TODO: should be O(update size)
      for i,uid in enumerate(self._gt_dict.keys()):
        self._gt_vec[i] = self._gt_dict[uid]
    # Both/neither idxs and uids defined    
    else:
      raise ValueError("Exactly one of idxs and uids must be not None")
    # Update dev sets
    self._update_devs(self.dev_split)
    
  def get_labeled_ground_truth(self, subset=None):
    """ Get indices and labels of subset which have ground truth """
    gt_all = self._gt_vec
    if subset is None:
      has_gt = (gt_all != 0)
      return np.ravel(np.where(has_gt)), gt_all[has_gt]
    if subset is 'training':
      t = self.training
      gt_all = gt_all[t]
      has_gt = (gt_all != 0)
      return t[has_gt], gt_all[has_gt]
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
      raise ValueError("subset must be 'training', 'test', 'validation' or an\
                        array of indices 0 <= i < {}".format(self.n()))

class DDLiteModel:
  def __init__(self, candidates, feats=None, gt_dict=None):
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
    # LF data
    self.lf_matrix = None
    self.lf_names = []
    # Model data    
    self.X = None
    self._w_fit = None
    self.w = None
    # GT data
    self.gt = CandidateGT(candidates, gt_dict)
    # MT data
    self._current_mindtagger_samples = np.array([], dtype=int)
    self._mt_tags = []
    self.mindtagger_instance = None
    # Status
    self.use_lfs = True
    self.model = None
    
  #########################################################
  #################### Basic size info ####################
  #########################################################

  def num_candidates(self):
    return len(self.C)
    
  def num_feats(self):
    return self.feats.shape[1]
  
  def num_lfs(self, result='all'):
    if self.lf_matrix is None:
      return 0
    return self.lf_matrix.shape[1]
    
  #######################################################
  #################### GT attributes ####################
  #######################################################
    
  def gt_dictionary(self):
    return self.gt._gt_dict
    
  def holdout(self):
    return self.gt.holdout()
    
  def validation(self):
    return self.gt.validation
    
  def test(self):
    return self.gt.test

  def dev(self):
    return self.gt.dev()

  def dev1(self):
    return self.gt.dev1

  def dev2(self):
    return self.gt.dev2
    
  def training(self):
    return self.gt.training
    
  def set_holdout(self, idxs=None, validation_frac=0.5):
    self.gt.set_holdout(idxs, validation_frac)
    
  def get_labeled_ground_truth(self, subset=None):
    return self.gt.get_labeled_ground_truth(subset)
    
  def update_gt(self, gt, idxs=None, uids=None):
    self.gt.update_gt(gt, idxs, uids)
    
  def dev_size_warn(self):
    self.gt.dev_size_warn()

  def test_val_size_warn(self):
    self.gt.val_test_size_warn()
    
  def get_gt_dict(self):
    return self.gt.get_gt_dict()    
    
  #######################################################
  #################### LF operations ####################
  #######################################################
                       
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
    
  #######################################################
  #################### LF stat comp. ####################
  #######################################################    

  def _cover(self, idxs=None):
    idxs = self.training() if idxs is None else idxs
    return [np.ravel((self.lf_matrix[idxs,:] == lab).sum(1))
            for lab in [1,-1]]

  def coverage(self, cov=None, idxs=None):
    cov = self._cover(idxs) if cov is None else cov    
    return np.mean((cov[0] + cov[1]) > 0)

  def overlap(self, cov=None, idxs=None):    
    cov = self._cover(idxs) if cov is None else cov    
    return np.mean((cov[0] + cov[1]) > 1)

  def conflict(self, cov=None, idxs=None):    
    cov = self._cover(idxs) if cov is None else cov    
    return np.mean(np.multiply(cov[0], cov[1]) > 0)

  def print_lf_stats(self, idxs=None):
    """
    Returns basic summary statistics of the LFs on training set (default) or
    passed idxs
    * Coverage = % of candidates that have at least one label
    * Overlap  = % of candidates labeled by > 1 LFs
    * Conflict = % of candidates with conflicting labels
    """
    cov = self._cover(idxs)
    print "LF stats on training set" if idxs is None else "LF stats on idxs"
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
    plt.title("(a) Label balance (training set coverage: {:.2f}%)".format(tot_cov))
    # LF conflict
    plt.subplot(1,n_plots,2)
    tot_conf = self._plot_conflict(cov)
    plt.title("(b) Label heat map (training set conflict: {:.2f}%)".format(tot_conf))
    # Show plots    
    plt.show()

  def _lf_conf(self, lf_idx):
    lf_csc = self.lf_matrix.tocsc()
    other_idx = np.concatenate((range(lf_idx),range(lf_idx+1, self.num_lfs())))
    ts = self.training()
    agree = lf_csc[:, other_idx].multiply(lf_csc[:, lf_idx])
    agree = agree[ts,:]
    return float((np.ravel((agree == -1).sum(1)) > 0).sum()) / len(ts)
    
  def top_conflict_lfs(self, n=10):
    """ Show the LFs with the highest mean conflicts per candidate """
    d = {nm : ["{:.2f}%".format(100.*self._lf_conf(i))]
         for i,nm in enumerate(self.lf_names)}
    tab = DictTable(sorted(d.items(), key=lambda t:t[1], reverse=True))
    tab.set_rows(n)
    tab.set_cols(2)
    tab.set_title(["Labeling function", "Percent candidates where LF has conflict"])
    return tab
    
  def _lf_coverage(self, lf_idx):
    lf_v = np.ravel(self.lf_matrix.tocsc()[self.training(), lf_idx].todense())
    return 1 - np.mean(lf_v == 0)
    
  def lowest_coverage_lfs(self, n=10):
    """ Show the LFs with the highest fraction of abstains """
    d = {nm : ["{:.2f}%".format(100.*self._lf_coverage(i))]
         for i,nm in enumerate(self.lf_names)}
    tab = DictTable(sorted(d.items(), key=lambda t:t[1]))
    tab.set_rows(n)
    tab.set_cols(2)
    tab.set_title(["Labeling function", "Candidate coverage"])
    return tab

  def _lf_acc(self, subset, lf_idx):
    gt = self.gt._gt_vec
    pred = np.ravel(self.lf_matrix.tocsc()[:,lf_idx].todense())
    has_label = np.where(pred != 0)
    has_gt = np.where(gt != 0)
    # Get labels/gt for candidates in dev set, with label, with gt
    gd_idxs = np.intersect1d(has_label, subset)
    gd_idxs = np.intersect1d(has_gt, gd_idxs)
    gt = np.ravel(gt[gd_idxs])
    pred_sub = np.ravel(pred[gd_idxs])
    n_neg = np.sum(pred_sub == -1)
    n_pos = np.sum(pred_sub == 1)
    if np.sum(pred == -1) == 0:
      neg_acc = -1
    elif n_neg == 0:
      neg_acc = 0
    else:
      neg_acc = float(np.sum((pred_sub == -1) * (gt == -1))) / n_neg
    if np.sum(pred == 1) == 0:
      pos_acc = -1
    elif n_pos == 0:
      pos_acc = 0
    else: 
      pos_acc = float(np.sum((pred_sub == 1) * (gt == 1))) / n_pos
    return (pos_acc, n_pos, neg_acc, n_neg)
    
  def _lf_acc_gen(self, lf_idx):
    pos_acc1, n_pos, neg_acc1, n_neg = self._lf_acc(self.dev1(), lf_idx)
    pos_acc2, n_pos2, neg_acc2, n_neg2 = self._lf_acc(self.dev2(), lf_idx)
    pos_acc2, neg_acc2 = max(0, pos_acc2), max(0, neg_acc2)
    return (pos_acc1, n_pos, abs(pos_acc1 - pos_acc2), n_pos2,
            neg_acc1, n_neg, abs(neg_acc1 - neg_acc2), n_neg2)    
    
  def lowest_empirical_accuracy_lfs(self, n=10):
    self.dev_size_warn()
    print "100% accuracy and 0 generalization score are \"perfect\""
    """ Show the LFs with the lowest accuracy compared to ground truth """
    d = {nm : list(self._lf_acc_gen(i)) for i,nm in enumerate(self.lf_names)}
    tab_pos = DictTable(sorted(d.items(), key=lambda t:t[1][0]))
    for k in tab_pos:
      if tab_pos[k][0] < 0:
        del tab_pos[k]
        continue
      tab_pos[k] = ["{:.2f}% (n={})".format(100.*tab_pos[k][0], tab_pos[k][1]),
                    "{:.2f} (n={})".format(tab_pos[k][2], tab_pos[k][3])] 
    tab_pos.set_rows(n)
    tab_pos.set_cols(3)
    tab_pos.set_title(["Labeling function", "Positive accuracy",
                       "Gen. score"])
    
    tab_neg = DictTable(sorted(d.items(), key=lambda t:t[1][4]))
    for k in tab_neg:
      if tab_neg[k][4] < 0:
        del tab_neg[k]
        continue
      tab_neg[k] = ["{:.2f}% (n={})".format(100.*tab_neg[k][4], tab_neg[k][5]),
                    "{:.2f} (n={})".format(tab_neg[k][6], tab_neg[k][7])]
    tab_neg.set_rows(n)
    tab_neg.set_cols(3)
    tab_neg.set_title(["Labeling function", "Negative accuracy",
                       "Gen. score"])
    return SideTables(tab_pos, tab_neg)
    
  def lf_summary_table(self):
    d = {nm : [self._lf_coverage(i), self._lf_conf(i), self._lf_acc_gen(i)]
         for i,nm in enumerate(self.lf_names)}
    for k,v in d.items():
      del d[k]
      pos_k, both_k = (v[2][0] >= 0), (v[2][0] >= 0 and v[2][4] >= 0)
      col, tp, pa, pg, na, ng = ("#ee0b40", "Negative", "N/A", "N/A",
                                 "{:.2f}% (n={})".format(100.*v[2][4], v[2][5]),
                                 "{:.2f} (n={})".format(v[2][6], v[2][7]))
      if pos_k:
        col, tp, na, ng, pa, pg = ("#0099ff", "Positive", "N/A", "N/A",
                                   "{:.2f}% (n={})".format(100.*v[2][0], v[2][1]),
                                   "{:.2f} (n={})".format(v[2][2], v[2][3]))
      if both_k:
        col, tp, pa, pg, na, ng = ("#c700ff", "Both",
                                   "{:.2f}% (n={})".format(100.*v[2][0], v[2][1]),
                                   "{:.2f} (n={})".format(v[2][2], v[2][3]),
                                   "{:.2f}% (n={})".format(100.*v[2][4], v[2][5]),
                                   "{:.2f} (n={})".format(v[2][6], v[2][7]))     
      fancy_k = "<b><font color=\"{}\">{}</font></b>".format(col, k)
      d[fancy_k] = [tp, "{:.2f}%".format(100.*v[0]),
                      "{:.2f}%".format(100.*v[1]), pa, pg, na, ng]
    tab = DictTable(sorted(d.items(), key=lambda t:t[1][0]))
    tab.set_rows(len(self.lf_names))
    tab.set_cols(8)
    tab.set_title(["Labeling<br />function", "Label<br />type",
                   "Candidate<br />coverage", "Candidate<br />conflict", 
                   "Positive<br />accuracy", "Positive<br />gen. score",
                   "Negative<br />accuracy", "Negative<br />gen. score"])
    return tab
    

  ######################################################
  #################### Learning fns ####################
  ######################################################

  def learn_weights_validated(self, **kwargs):
    if len(self.validation) == 0:
      raise ValueError("No validation set. Use set_holdout(p) with p>0.")
    return self.learn_weights(**kwargs)

  def learn_weights(self, n_iter=1000, tol=1e-6, sample=False, 
                    n_samples=100, mu=None, n_mu=20, mu_min_ratio=1e-6, 
                    alpha=0, rate=0.01, decay=0.99, bias=False, 
		            warm_starts=False, use_sparse=True, verbose=False,
                    log=True, plot=True):
    """
    Uses the N x R matrix of LFs and the N x F matrix of features
    Stacks them, giving the LFs a +1 prior (i.e. init value)
    Then runs learning, saving the learned weights
    Holds out preset set of candidates for evaluation
    """
    self.test_val_size_warn()
    self.model = "Joint"
    N, R, F = self.num_candidates(), self.num_lfs(), self.num_feats()
    self.X = sparse.hstack([self.lf_matrix, self.feats], format='csr') if not bias\
	       else sparse.hstack([self.lf_matrix, self.feats, np.ones((N,1))], format='csr')
    if not use_sparse:
      self.X = np.asarray(self.X.todense())
    w0 = np.concatenate([np.ones(R), np.zeros(F)]) if not bias\
 	   else np.concatenate([np.ones(R), np.zeros(F+1)])
    unreg = [self.X.shape[1]-1] if bias else []
    # If a single mu is provided, just fit a single model
    if mu is not None and (not hasattr(mu, '__iter__') or len(mu) == 1):
      mu = mu if not hasattr(mu, '__iter__') else mu[0]
      self.w = learn_elasticnet_logreg(self.X[self.training(),:],
                                       n_iter=n_iter, tol=tol, w0=w0, 
                                       mu_seq=mu, alpha=alpha, sample=sample,
                                       n_samples=n_samples, warm_starts=warm_starts,
                                       rate=rate, decay=decay, unreg=unreg,
                                       verbose=verbose)[mu]
    # TODO: handle args between learning functions better
    elif len(self.validation()) > 0: 
      result = learn_elasticnet_logreg(self.X[self.training(),:],
                                       n_iter=n_iter, tol=tol, w0=w0, n_mu=n_mu, 
                                       mu_seq=mu, mu_min_ratio=mu_min_ratio,
                                       alpha=alpha, rate=rate, decay=decay, 
                                       sample=sample, n_samples=n_samples,
                                       warm_starts=warm_starts, unreg=unreg,
                                       verbose=verbose)
      self._w_fit = OrderedDict()
      gt = self.gt._gt_vec[self.validation()]
      f1_opt, w_opt, mu_opt = 0, None, None
      s = 0 if self.use_lfs else self.num_lfs()
      for mu in sorted(result.keys()):
        w = result[mu]
        pred = 2*(odds_to_prob(self.X[self.validation(),s:].dot(w[s:])) >= 0.5) - 1
        prec, rec = precision(gt, pred), recall(gt, pred)
        f1 = f1_score(prec = prec, rec = rec)
        self._w_fit[mu] = ValidatedFit(w, prec, rec, f1)
        if f1 >= f1_opt:
          w_opt, f1_opt, mu_opt = w, f1, mu
      self.w = w_opt
      if plot:
        self.plot_learning_diagnostics(mu_opt, f1_opt)
    # Fit using a default mu value  
    else:
      warnings.warn("Using default mu value with no validation set")
      mu = 1e-7
      self.w = learn_elasticnet_logreg(self.X[self.training(),:],
                                       n_iter=n_iter, tol=tol, w0=w0, 
                                       mu_seq=mu, alpha=alpha, sample=sample,
                                       n_samples=n_samples, warm_starts=warm_starts,
                                       rate=rate, decay=decay, unreg=unreg,
                                       verbose=verbose)[mu]
    if log:
      return self.add_to_log()
      
  def plot_learning_diagnostics(self, mu_opt, f1_opt):
    
    mu_seq = sorted(self._w_fit.keys())
    p = np.ravel([self._w_fit[mu].P for mu in mu_seq])
    r = np.ravel([self._w_fit[mu].R for mu in mu_seq])
    f1 = np.ravel([self._w_fit[mu].F1 for mu in mu_seq])
    nnz = np.ravel([np.sum(self._w_fit[mu].w != 0) for mu in mu_seq])    
    
    fig, ax1 = plt.subplots()
    # Plot spread
    ax1.set_xscale('log', nonposx='clip')    
    ax1.scatter(mu_opt, f1_opt, marker='*', color='purple', s=500,
                zorder=10, label="Maximum F1: mu={}".format(mu_opt))
    ax1.plot(mu_seq, f1, 'o-', color='red', label='F1 score')
    ax1.plot(mu_seq, p, 'o--', color='blue', label='Precision')
    ax1.plot(mu_seq, r, 'o--', color='green', label='Recall')
    ax1.set_xlabel('log(penalty)')
    ax1.set_ylabel('F1 score/Precision/Recall')
    ax1.set_ylim(-0.04, 1.04)
    for t1 in ax1.get_yticklabels():
      t1.set_color('r')
    # Plot nnz
    ax2 = ax1.twinx()
    ax2.plot(mu_seq, nnz, '.:', color='gray', label='Sparsity')
    ax2.set_ylabel('Number of non-zero coefficients')
    ax2.set_ylim(-0.01*np.max(nnz), np.max(nnz)*1.01)
    for t2 in ax2.get_yticklabels():
      t2.set_color('gray')
    # Shrink plot for legend
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0+box1.height*0.1, box1.width, box1.height*0.9])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0+box2.height*0.1, box2.width, box2.height*0.9])
    plt.title("Validation for logistic regression learning")
    lns1, lbs1 = ax1.get_legend_handles_labels()
    lns2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1+lns2, lbs1+lbs2, loc='upper center', bbox_to_anchor=(0.5,-0.05),
               scatterpoints=1, fontsize=10, markerscale=0.5)
    plt.show()
      
  def set_use_lfs(self, use=True):
    self.use_lfs = bool(use)

  def get_log_odds(self, subset=None):
    """
    Get the array of predicted link function values (continuous) given weights
    Return either all candidates, a specified subset, or only validation/test set
    """
    start = 0 if self.use_lfs else self.num_lfs()
    if self.X is None or self.w is None:
      raise ValueError("Inference has not been run yet")
    if subset is None:
      return self.X[:, start:].dot(self.w[start:])
    if subset is 'test':
      return self.X[self.test(), start:].dot(self.w[start:])
    if subset is 'validation':
      return self.X[self.validation(), start:].dot(self.w[start:])
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
    return odds_to_prob(self.get_log_odds(subset))
 
  def get_predicted(self, subset=None):
    """
    Get the array of predicted (boolean) variables given weights
    Return either all variables, a specified subset, or only validation/test set
    """
    return np.sign(self.get_log_odds(subset))

  def get_classification_accuracy(self, subset=None):
    """
    Given the ground truth, return the classification accuracy
    Return either accuracy for all candidates, a subset, or validation/test set
    """
    idxs, gt = self.get_labeled_ground_truth(subset)
    pred = self.get_predicted(idxs)
    return np.mean(gt == pred)
    
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
    idxs, gt = self.get_labeled_ground_truth(None)
    has_test, has_gt = (len(self.test()) > 0), (len(gt) > 0)
    n_plots = 1 + has_test + (has_test and has_gt)
    # Whole set histogram
    plt.subplot(1,n_plots,1)
    probs = self.get_predicted_probability()
    self._plot_prediction_probability(probs)
    plt.title("(a) # Predictions (whole set)")
    # Hold out histogram
    if has_test:
      plt.subplot(1,n_plots,2)
      self._plot_prediction_probability(probs[self.test()])
      plt.title("(b) # Predictions (test set)")
      # Classification bucket accuracy
      if has_gt:
        plt.subplot(1,n_plots,3)
        self._plot_accuracy(probs[idxs], gt)
        plt.title("(c) Accuracy (test set)")
    plt.show()
    
  def _get_all_abstained(self, training=True):
    idxs = self.training() if training else range(self.num_candidates())
    return np.ravel(np.where(np.ravel((self.lf_matrix[idxs,:]).sum(1)) == 0))

  def open_mindtagger(self, num_sample=None, abstain=False, **kwargs):
    self.mindtagger_instance = MindTaggerInstance(self.C.mindtagger_format())
    if isinstance(num_sample, int) and num_sample > 0:
      pool = self._get_all_abstained(dev=True) if abstain else self.training()
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
    tags_l = self.gt._gt_vec[self._current_mindtagger_samples]
    tags = np.zeros_like(self.gt._gt_vec)
    tags[self._current_mindtagger_samples] = tags_l
    return self.mindtagger_instance.open_mindtagger(self.C.generate_mindtagger_items,
                                                    self._current_mindtagger_samples,
                                                    probs, tags, **kwargs)
  
  def add_mindtagger_tags(self):
    tags = self.mindtagger_instance.get_mindtagger_tags()
    self._mt_tags = tags
    is_tagged = [i for i,tag in enumerate(tags) if 'is_correct' in tag]
    tb = [tags[i]['is_correct'] for i in is_tagged]
    tb = [1 if t else -1 for t in tb]
    idxs = self._current_mindtagger_samples[is_tagged]
    self.update_gt(tb, idxs=idxs)
  
  def add_to_log(self, log_id=None, subset='test', show=True):
    if log_id is None:
      log_id = len(self.logger)
    gt_idxs, gt = self.get_labeled_ground_truth(subset)
    pred = self.get_predicted(gt_idxs)    
    self.logger.log(ModelLog(log_id, self.lf_names, self.use_lfs, self.model,
                             gt_idxs, gt, pred))
    if show:
      return self.logger[-1]
      
  def show_log(self, idx=None):
    if idx is None:
      return self.logger
    try:
      return self.logger[idx]
    except:
      raise ValueError("Index must be for one of {} logs".format(len(self.logger)))

# Legacy name
CandidateModel = DDLiteModel

####################################################################
############################ ALGORITHMS ############################
#################################################################### 

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
