# Base Python
import atexit, bisect, json, os, pipes, re, socket, sys, warnings
from collections import defaultdict

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
from tree_structs import corenlp_to_xmltree
from entity_features import *

# ddlite parsers
from parser import *

##################################################################
############################ MATCHERS ############################
################################################################## 

class Matcher(object):
  def apply(self, s):
    raise NotImplementedError()
    
class DictionaryMatch(Matcher):
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

class RegexMatch(Matcher):
  """Selects according to ngram-matching against a regex """
  def __init__(self, label, regex_pattern, match_attrib='words', ignore_case=True):
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
    except TypeError:
      seq = s.__dict__[self.match_attrib]
    
    # Convert character index to token index
    start_c_idx = [0]
    for s in seq:
      start_c_idx.append(start_c_idx[-1]+len(s)+1)
    # Find regex matches over phrase
    phrase = ' '.join(seq)
    for match in self._re_comp.finditer(phrase):
      start = bisect.bisect(start_c_idx, match.start())
      end = bisect.bisect(start_c_idx, match.end())
      yield list(range(start-1, end))
      
class MultiMatcher(Matcher):
  """ 
  Wrapper to apply multiple matchers of a given entity type 
  Priority of labeling given by matcher order
  """
  def __init__(self, label, matchers):
    self.label = label
    self.matchers = matchers
  def apply(self, s):
    applied = set()
    for m in self.matchers:
      for rg in m.apply(s):
        if rg[0] not in applied:
          applied.add(rg[0])
          yield rg        
    

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
  
####################################################################
############################ MINDTAGGER ############################
#################################################################### 

class MindTaggerInstance:

  def __init__(self, mindtagger_format):
    self._mindtagger_format = mindtagger_format
    self.instance = None
    atexit.register(self._kill_mindtagger)
  
  def _system(self, script):
      return os.system("bash -c %s" % pipes.quote(script))
    
  def _kill_mindtagger(self):
    if self.instance is not None:
      self._system("kill -TERM $(cat mindtagger.pid)")
    
  def _generate_task_format(self):
    return  """
            <mindtagger mode="precision">

              <template for="each-item">
                %(title_block)s
                with probability <strong>{{item.probability | number:3}}</strong>
                appeared in sentence {{item.sent_id}} of document {{item.doc_id}}:
                <blockquote>
                    <big mindtagger-word-array="item.words" array-format="json">
                        %(style_block)s
                    </big>
                </blockquote>

                <div>
                  <div mindtagger-item-details></div>
                </div>
              </template>

              <template for="tags">
                <span mindtagger-adhoc-tags></span>
                <span mindtagger-note-tags></span>
              </template>

            </mindtagger>
            """ % self._mindtagger_format
    

  def launch_mindtagger(self, task_name, generate_items, task_root="mindtagger",
                        task_recreate = True, **kwargs):
                            
    args = dict(
        task = task_name,
        task_root = task_root,
        # figure out hostname and task name from IPython notebook
        host = socket.gethostname(),
        # determine a port number based on user name
        port = hash(os.getlogin()) % 1000 + 8000,
      )
    args['task_path'] = "%s/%s" % (args['task_root'], args['task'])
    args['mindtagger_baseurl'] = "http://%(host)s:%(port)s/" % args
    args['mindtagger_url'] = "%(mindtagger_baseurl)s#/mindtagger/%(task)s" % args
    # quoted values for shell
    shargs = { k: pipes.quote(str(v)) for k,v in args.iteritems() }

    

    # install mindbender included in DeepDive's release
    print "Making sure MindTagger is installed. Hang on!"
    if self._system("""
      export PREFIX="$PWD"/deepdive
      [[ -x "$PREFIX"/bin/mindbender ]] ||
      bash <(curl -fsSL git.io/getdeepdive || wget -qO- git.io/getdeepdive) deepdive_from_release
    """ % shargs) != 0: raise OSError("Mindtagger could not be installed")

    if task_recreate or not os.path.exists(args['task_path']):
      # prepare a mindtagger task from the data this object is holding
      try:
        if self._system("""
          set -eu
          t=%(task_path)s
          mkdir -p "$t"
          """ % shargs) != 0: raise OSError("Mindtagger task could not be created")
        with open("%(task_path)s/mindtagger.conf" % args, "w") as f:
          f.write("""
            title: %(task)s Labeling task for estimating precision
            items: {
                file: items.csv
                key_columns: [ext_id]
            }
            template: template.html
            """ % args)
        with open("%(task_path)s/template.html" % args, "w") as f:
          f.write(self._generate_task_format())
        with open("%(task_path)s/items.csv" % args, "w") as f:
          # prepare data to label
          import csv
          items = generate_items()
          item = next(items)
          o = csv.DictWriter(f, fieldnames=item.keys())
          o.writeheader()
          o.writerow(item)
          for item in items:
            o.writerow(item)
      except:
        raise OSError("Mindtagger task data could not be prepared: %s" % str(sys.exc_info()))

    # launch mindtagger
    if self._system("""
      # restart any running mindtagger instance
      ! [ -s mindtagger.pid ] || kill -TERM $(cat mindtagger.pid) || true
      PORT=%(port)s deepdive/bin/mindbender tagger %(task_root)s/*/mindtagger.conf &
      echo $! >mindtagger.pid
      """ % shargs) != 0: raise OSError("Mindtagger could not be started")
    while self._system("curl --silent --max-time 1 --head %(mindtagger_url)s >/dev/null" % shargs) != 0:
      time.sleep(0.1)        

    self.instance = args
    return args['mindtagger_url']
  
  def open_mindtagger(self, generate_mt_items, num_sample = 100, **kwargs):

    def generate_items():
      return generate_mt_items(num_sample)      

    # determine a task name using hash of the items
    # See: http://stackoverflow.com/a/7823051/390044 for non-negative hexadecimal
    def tohex(val, nbits):
      return "%x" % ((val + (1 << nbits)) % (1 << nbits))
    task_name = tohex(hash(json.dumps([i for i in generate_items()])), 64)

    # launch Mindtagger
    url = self.launch_mindtagger(task_name, generate_items, **kwargs)

    # return an iframe
    from IPython.display import IFrame
    return IFrame(url, **kwargs)
    
  def get_mindtagger_tags(self):
    tags_url = "%(mindtagger_baseurl)sapi/mindtagger/%(task)s/tags.json?attrs=ext_id" % self.instance

    import urllib, json
    opener = urllib.FancyURLopener({})
    t = opener.open(tags_url)
    tags = json.loads(t.read())

    return tags

#####################################################################
############################ EXTRACTIONS ############################
#####################################################################

class Extraction(object):
  """ Proxy providing an interface into the Extractions class """
  def __init__(self, extractions, ex_id):
    self.extractions = extractions
    self.id = ex_id
  def __getattr__(self, name):
    if name == 'probability':
      if self.extractions.X is None or self.extractions.w is None:
        return None
      else:
        return self.extractions.get_predicted_probability(subset=[self.id])[0]
    return getattr(self.extractions._extractions[self.id], name)
  def __repr__(self):
    return str(self.extractions._extractions[self.id])

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
  
  def __repr__(self):
    raise NotImplementedError()
    
class Extractions(object):
  """
  Base class for a collection of extractions
  Sub-classes need to yield extractions from sentences (_apply) and 
  generate features (_get_features)
  See Relations and Entities for examples
  """
  def __init__(self, sents):
    """
    Set up learning problem and generate candidates
     * sents is a flat list of Sentence objects
    """
    self.rules = None
    self.feats = None
    self.X = None
    self.feat_index = {}
    self.w = None
    self.holdout = []
    self._extractions = list(self._extract(sents))
    self.mindtagger_instance = None
  
  def __getitem__(self, i):
    return Extraction(self, i)
    
  def __len__(self):
    return len(self._extractions)

  def __iter__(self):
    return (Extraction(self, i) for i in xrange(0, len(self)))
  
  def num_extractions(self):
    return len(self)
  
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
    
  def _extract(self, sents):
    for sent in sents:
      for ext in self._apply(sent):
        yield ext

  def _apply(self, sent):
    raise NotImplementedError()
    
  def apply_rules(self, rules_f, clear=False):
    """ Apply rule functions given in list
    Allows adding to existing rules or clearing rules with CLEAR=True
    """
    nr_old = self.num_rules() if not clear else 0
    add = sparse.lil_matrix((self.num_extractions(), len(rules_f)))
    self.rules = add if (self.rules is None or clear)\
                     else sparse.hstack([self.rules,add], format = 'lil')
    for i,ext in enumerate(self._extractions):    
      for ja,rule in enumerate(rules_f):
        self.rules[i,ja + nr_old] = rule(ext)
        
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
        
  def learn_feats_and_weights(self, nSteps=1000, sample=False, nSamples=100,
        mu=1e-9, holdout=0.1, use_sparse = True, verbose=False):
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
    self.w = learn_params(self.X[np.setdiff1d(range(N), self.holdout), :],
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
    
  def generate_mindtagger_items(self):
    raise NotImplementedError()
    
  def mindtagger_format(self):
    raise NotImplementedError()
    
  def open_mindtagger(self, num_sample = 100, **kwargs):
    self.mindtagger_instance = MindTaggerInstance(self.mindtagger_format())
    return self.mindtagger_instance.open_mindtagger(self.generate_mindtagger_items,
                                                    num_sample, **kwargs)
  
  def get_mindtagger_tags(self):
    return self.mindtagger_instance.get_mindtagger_tags()
  
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
  def __init__(self, e1, e2, sents):
    if not issubclass(e1.__class__, Matcher):
      warnings.warn("e1 is not a Matcher subclass")
    if not issubclass(e2.__class__, Matcher):
      warnings.warn("e2 is not a Matcher subclass")
    self.e1 = e1
    self.e2 = e2
    super(Relations, self).__init__(sents)
  
  def __getitem__(self, i):
    return Relation(self, i)  
  
  def _apply(self, sent):
    xt = corenlp_to_xmltree(sent)
    for e1_idxs in self.e1.apply(sent):
      for e2_idxs in self.e2.apply(sent):
        yield relation_internal(e1_idxs, e2_idxs, self.e1.label, 
                                self.e2.label, sent, xt)
  
  def _get_features(self, method='treedlib'):
    get_feats = compile_relation_feature_generator()
    f_index = defaultdict(list)
    for j,ext in enumerate(self._extractions):
      for feat in get_feats(ext.root, ext.e1_idxs, ext.e2_idxs):
        f_index[feat].append(j)
    return f_index
    
  def generate_mindtagger_items(self, n_samp):
    N = len(self)
    ext_sample = np.random.choice(N, n_samp, replace=False) if N > n_samp else range(N) 
    for i in ext_sample:
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
        probability     = item.probability
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
  def __init__(self, e, sents):
    if not issubclass(e.__class__, Matcher):
      warnings.warn("e is not a Matcher subclass")
    self.e = e
    super(Entities, self).__init__(sents)
  
  def __getitem__(self, i):
    return Entity(self, i)  
  
  def _apply(self, sent):
    xt = corenlp_to_xmltree(sent)
    for e_idxs in self.e.apply(sent):
      yield entity_internal(e_idxs, self.e.label, sent, xt)        
  
  def _get_features(self, method='treedlib'):
    get_feats = compile_entity_feature_generator()
    f_index = defaultdict(list)
    for j,ext in enumerate(self._extractions):
      for feat in get_feats(ext.root, ext.idxs):
        f_index[feat].append(j)
      for feat in get_ddlib_feats(ext, ext.idxs):
        f_index["DDLIB_" + feat].append(j)
    return f_index    
  
  def generate_mindtagger_items(self, n_samp):
    N = self.num_extractions()
    ext_sample = np.random.choice(N, n_samp, replace=False) if N > n_samp else range(N) 
    for i in ext_sample:
      item = self[i]      
      yield dict(
        ext_id          = item.id,
        doc_id          = item.doc_id,
        sent_id         = item.sent_id,
        words           = json.dumps(item.words),
        idxs            = json.dumps(item.idxs),
        label           = item.label,
        probability     = item.probability
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

def learn_params(X, nSteps, w0=None, sample=True, nSamples=100, mu=1e-9, verbose=False):
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

  print "***** Relation0 *****"
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
  DR = Entities(MultiMatcher('COMBO', [b,vbz]), sents)
  for dr in DR:
      print dr.tagged_sent

if __name__ == '__main__':
  main()
