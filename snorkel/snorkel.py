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
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree, XMLTree
from entity_features import *

# ddlite parsers
from parser import *

# ddlite matchers
from matchers import *

# ddlite mindtagger
from mindtagger import *

# ddlite learning
from learning import learn_elasticnet_logreg, odds_to_prob, get_mu_seq,\
                            DEFAULT_RATE, DEFAULT_MU, DEFAULT_ALPHA
# ddlite LSTM
from lstm import *

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
      with open(C, 'rb') as f:
        self._candidates = cPickle.load(f)
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
    

class Learner(object):
    """
    Core learning class for Snorkel, encapsulating the overall pipeline:
        1. Generating a noisy training set by applying the LFs to the candidates
        2. Modeling this noisy training set
        3. Learning a model over the candidate features, trained on the noisy training set

    As input takes:
        - A set of Candidate objects
        - A set of labeling functions (LFs) which are functions f : Candidate -> {-1,0,1}
        - A Featurizer object, which is applied to the Candidate objects to generate features
        - A Model object, representing the model to train
        - _A test set to compute performance against, which consists of a dict mapping candidate id -> T/F_
    """
    # TODO: Tuner (GridSearch) class that wraps this! 
    #def __init__(self, candidates, lfs, model=LogReg, featurizer=None, test_set=None):
    def __init__(self, candidates, lfs, model=None, featurizer=None, test_set=None):
        
        # (1) Generate the noisy training set T by applying the LFs
        print "Applying LFs to Candidates..."
        self.candidates = candidates
        self.lfs        = lfs

        # T_{i,j} is the label (in {-1,0,1}) given to candidate i by LF j
        self.T = self._apply_lfs(self)

        # Generate features; F_{i,j} is 1 if candidate i has feature j
        if featurizer is not None:
            print "Feauturizing candidates..."
            self.F = featurizer.apply(self.candidates)
        self.model    = model
        self.test_set = test_set
        self.X        = None
        self.w0       = None
        self.w        = None

    def _apply_lfs(self):
        """Apply the labeling functions to the candidates to populate X"""
        # TODO: Parallelize this
        self.X = sparse.lil_matrix((len(self.candidates), len(lfs)))
        for i,c in enumerate(self.candidates):
            for j,lf in enumerate(self.lfs):
                self.X[i,j] = lf(c)
        self.X = self.X.tocsr()

    def _print_test_score():
        # TODO
        raise NotImplementedError()

    def train_model(self, feat_w0=0.0, lf_w0=1.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        # TODO: Bias term
        n, m = self.T.shape
        _, f = self.F.shape
        if self.X is None:
            self.X  = sparse.hstack([self.T, self.F], format='csc')

        # Set initial values for feature weights
        self.w0 = np.concatenate([lf_w0*np.ones(m), feat_w0*np.ones(f)])

        # Train model
        self.w = self.model(self.X, w0=self.w0, **model_hyperparams).train()
        
        # Print out score if test_set was provided
        self._print_test_score()


class PipelinedLearner(Learner):
    """Implements the **"pipelined" approach**"""
    def train_model(self, feat_w0=0.0, lf_w0=1.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        # TODO: Bias term
        m, n = self.T.shape
        f, _ = self.F.shape

        # Learn lf accuracies first
        # TODO: Distinguish between model hyperparams...
        self.lf_accs = LogReg(self.T, w0=lf_w0*np.ones(m), **model_hyperparams)

        # Compute marginal probabilities over the candidates from this model of the training set
        # TODO

        # Learn model over features
        #self.w = self.model( \
        #    self.F, training_marginals=self.training_marginals, w0=feat_w0*np.ones(f), **model_hyperparams)

        # Print out score if test_set was provided
        #self._print_test_score()
