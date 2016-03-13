import os, sys, json
from collections import namedtuple, defaultdict
import random
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from tree_structs import corenlp_to_xmltree

sys.path.append('{}/treedlib'.format(os.getcwd()))
from treedlib import compile_relation_feature_generator
from entity_features import *

from parser import *

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
  
class MindTaggerInstance:

  def __init__(self, mindtagger_format):
    self._mindtagger_format = mindtagger_format
    self.instance = None
    
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
    import socket, pipes

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

    def system(script):
      return os.system("bash -c %s" % pipes.quote(script))

    # install mindbender included in DeepDive's release
    print "Making sure MindTagger is installed. Hang on!"
    if system("""
      export PREFIX="$PWD"/deepdive
      [[ -x "$PREFIX"/bin/mindbender ]] ||
      bash <(curl -fsSL git.io/getdeepdive || wget -qO- git.io/getdeepdive) deepdive_from_release
    """ % shargs) != 0: raise OSError("Mindtagger could not be installed")

    if task_recreate or not os.path.exists(args['task_path']):
      # prepare a mindtagger task from the data this object is holding
      try:
        if system("""
          set -eu
          t=%(task_path)s
          mkdir -p "$t"
          """ % shargs) != 0: raise OSError("Mindtagger task could not be created")
        with open("%(task_path)s/mindtagger.conf" % args, "w") as f:
          f.write("""
            title: %(task)s Labeling task for estimating precision
            items: {
                file: items.csv
                key_columns: [sent_id]
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
    if system("""
      # restart any running mindtagger instance
      ! [ -s mindtagger.pid ] || kill -TERM $(cat mindtagger.pid) || true
      PORT=%(port)s deepdive/bin/mindbender tagger %(task_root)s/*/mindtagger.conf &
      echo $! >mindtagger.pid
      """ % shargs) != 0: raise OSError("Mindtagger could not be started")
    while system("curl --silent --max-time 1 --head %(mindtagger_url)s >/dev/null" % shargs) != 0:
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
    tags_url = "%(mindtagger_baseurl)sapi/mindtagger/%(task)s/tags.json?attrs=sent_id" % self.instance

    import urllib, json
    opener = urllib.FancyURLopener({})
    t = opener.open(tags_url)
    tags = json.loads(t.read())

    return tags

class Extraction(object):
    
  def __init__(self, all_idxs, labels, sent, xt):
    self.all_idxs = all_idxs
    self.labels = labels
    # Absorb XMLTree and Sentence object attributes for access by rules
    self.xt = xt
    self.root = self.xt.root
    self.__dict__.update(sent.__dict__)
    self.probability = None

    # Add some additional useful attributes
    self.tagged_sent = ' '.join(tag_seqs(self.words, self.all_idxs, self.labels))

  def render(self):
    self.xt.render_tree(self.all_idxs)

  def __repr__(self):
    raise NotImplementedError()
  

class Extractions(object):
  """
  Base class for extractions
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
    self.extractions = list(self._extract(sents))
    self.mindtagger_instance = None
    
  def _extract(self, sents):
    for sent in sents:
      for ext in self._apply(sent):
        yield ext

  def _apply(self, sent):
    raise NotImplementedError()
    
  def apply_rules(self, rules):
    self.rules = np.zeros((len(rules), len(self.extractions)))
    for i,rule in enumerate(rules):
      for j,ext in enumerate(self.extractions):
        self.rules[i,j] = rule(ext)
        
  def _get_features(self):
      raise NotImplementedError()    
    
  def extract_features(self, *args):
    f_index = self._get_features(args)
    # Apply the feature generator, constructing a sparse matrix incrementally
    # Note that lil_matrix should be relatively efficient as we proceed row-wise
    F = lil_matrix((len(f_index), len(self.extractions)))
    for i,feat in enumerate(f_index.keys()):
      self.feat_index[i] = feat
      for j in f_index[feat]:
        F[i,j] = 1
    self.feats = csr_matrix(F)

  def learn_feats_and_weights(self, nSteps=1000, sample=False, nSamples=100, mu=1e-9, holdout=0.1, verbose=False):
    """
    Uses the R x N matrix of rules and the F x N matrix of features defined
    for the Relations object
    Stacks them, giving the rules a +1 prior (i.e. init value)
    Then runs learning, saving the learned weights
    Holds out a set of variables for testing, either a random fraction or a specific set of indices
    """   
    R, N = self.rules.shape  # dense
    F, N = self.feats.shape  # sparse
    
    if hasattr(holdout, "__iter__"):
        self.holdout = holdout
    elif not hasattr(holdout, "__iter__") and (0 <= holdout < 1):
        self.holdout = np.random.choice(N, np.floor(holdout * N), replace=False)
    else:
        raise ValueError("Holdout must be an array of indices or fraction")    
    
    self.X = np.array(np.vstack([self.rules, self.feats.todense()]))
    w0 = np.concatenate([np.ones(R), np.zeros(F)])
    self.w = learn_params(self.X[:, np.setdiff1d(range(N), self.holdout)],
                          nSteps=nSteps, w0=w0, sample=sample,
                          nSamples=nSamples, mu=mu, verbose=verbose)
    probs = self.get_predicted_probability()
    for i, ext in enumerate(self.extractions):
      ext.probability = probs[i]

  def get_link(self, holdout_only=False):
    """
    Get the array of predicted link function values (continuous) given learned weight param w
    Return either all variables or only holdout set
    """
    return np.dot(self.X[:,self.holdout].T, self.w) if holdout_only else np.dot(self.X.T, self.w)       


  def get_predicted_probability(self, holdout_only=False):
    """
    Get the array of predicted probabilities (continuous) for variables given learned weight param w
    Return either all variables or only holdout set
    """
    return odds_to_prob(self.get_link(holdout_only))
 
  def get_predicted(self, holdout_only=False):
    """
    Get the array of predicted (boolean) variables given learned weight param w
    Return either all variables or only holdout set
    """
    return np.sign(self.get_link(holdout_only))
    
  def _handle_ground_truth(self, ground_truth, holdout_only):
    gt = None
    N = self.rules.shape[1]
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
    Note: ground_truth must either be an array the length of the full dataset, or of the holdout
          If the latter, holdout_only must be set to True
    """
    gt = self._handle_ground_truth(ground_truth, holdout_only)
    pred = self.get_predicted(holdout_only)
    return (np.dot(pred, gt) / len(gt) + 1) / 2

  def get_rule_priority_vote_accuracy(self, ground_truth, holdout_only=False):
    """
    This is to answer the question: 'How well would my rules alone do?'
    I.e. without any features, learning of rule or feature weights, etc.- this serves as a
    natural baseline / quick metric
    Labels are assigned by the first rule that emits one for each relation (based on the order
    of the provided rules list)
    Note: ground_truth must either be an array the length of the full dataset, or of the holdout
          If the latter, holdout_only must be set to True
    """
    R, N = self.rules.shape
    gt = self._handle_ground_truth(ground_truth, holdout_only)
    grid = self.holdout if holdout_only else xrange(N)
    correct = 0
    for j in grid:
      for i in xrange(R):
        if self.rules[i,j] != 0:
          correct += 1 if self.rules[i,j] == gt[j] else 0
          break
    return float(correct) / len(gt)
    
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
    return '\n'.join(str(e) for e in self.extractions)
  
          
class Relation(Extraction):
  def __init__(self, e1_idxs, e2_idxs, e1_label, e2_label, sent, xt):
    self.e1_idxs = e1_idxs
    self.e2_idxs = e2_idxs
    self.e1_label = e1_label
    self.e2_label = e2_label
    super(Relation, self).__init__([self.e1_idxs, self.e2_idxs],
                                   [self.e1_label, self.e2_label], sent, xt)

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
    self.relations = self.extractions

  def _apply(self, sent):
    xt = corenlp_to_xmltree(sent)
    for e1_idxs in self.e1.apply(sent):
      for e2_idxs in self.e2.apply(sent):
        yield Relation(e1_idxs, e2_idxs, self.e1.label, self.e2.label, sent, xt)
  
  def _get_features(self, method='treedlib'):
    get_feats = compile_relation_feature_generator()
    f_index = defaultdict(list)
    for j,ext in enumerate(self.extractions):
      for feat in get_feats(ext.root, ext.e1_idxs, ext.e2_idxs):
        f_index[feat].append(j)
    return f_index
    
  def generate_mindtagger_items(self, n_samp):
    N = len(self.extractions)
    ext_sample = random.sample(self.extractions, n_samp) if N > n_samp else self.extractions 
    for item in ext_sample:
      yield dict(
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
                        
    

class Entity(Extraction):
  def __init__(self, idxs, label, sent, xt):
    self.idxs = idxs
    self.label = label
    super(Entity, self).__init__([idxs], [label], sent, xt)

  def __repr__(self):
    return '<Entity: {}{}>'.format([self.words[i] for i in self.idxs], self.idxs)


class Entities(Extractions):
  def __init__(self, e, sents):
    if not issubclass(e.__class__, Matcher):
      warnings.warn("e is not a Matcher subclass")
    self.e = e
    super(Entities, self).__init__(sents)
    self.entities = self.extractions

  def _apply(self, sent):
    xt = corenlp_to_xmltree(sent)
    for e_idxs in self.e.apply(sent):
      yield Entity(e_idxs, self.e.label, sent, xt)        
  
  def _get_features(self, method='treedlib'):
    get_feats = compile_entity_feature_generator()
    f_index = defaultdict(list)
    for j,ext in enumerate(self.extractions):
      for feat in get_feats(ext.root, ext.idxs):
        f_index[feat].append(j)
      for feat in get_ddlib_feats(ext, ext.idxs):
        f_index["DDLIB_" + feat].append(j)
    return f_index    
  
  def generate_mindtagger_items(self, n_samp):
    N = len(self.extractions)
    ext_sample = random.sample(self.extractions, n_samp) if N > n_samp else self.extractions 
    for item in ext_sample:
      yield dict(
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

def main():
  txt = "Han likes Luke and a wookie. Han Solo don\'t like bounty hunters."
  parser = SentenceParser()
  sents = list(parser.parse(txt))

  g = DictionaryMatch('G', ['Han Solo', 'Luke', 'wookie'])
  b = DictionaryMatch('B', ['Bounty Hunters'])

  print "***** Relation0 *****"
  R = Relations(g, b, sents)
  print R
  print R.relations[0].tagged_sent
  
  print "***** Relation 1 *****"
  R = Relations(g, g, sents)
  print R
  for r in R.relations:
      print r.tagged_sent
  
  print "***** Entity *****"
  E = Entities(g, sents)
  print E                
  for e in E.entities:
      print e.tagged_sent

if __name__ == '__main__':
  main()
