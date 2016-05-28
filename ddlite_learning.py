import numpy as np
import scipy.sparse as sparse
import warnings

class joint_learning_opts:
  def __init__(self, kwargs):
    self.X, self.F, self.LF = None, None, None
    if 'X' in kwargs:
      self.X = kwargs['X']
      if type(self.X) != np.ndarray and not sparse.issparse(self.X):
        raise TypeError("Inputs should be np.ndarray type or scipy sparse.")
    else:
      F, LF = kwargs.get('F', None), kwargs.get('LF', None)
      if F is None or LF is None:
        raise ValueError("Must provide X matrix, or F and LF matrices")
      try:
        self.X = sparse.hstack([LF, F], format='csr')
      except Exception as e:
        raise ValueError("Could not join F and LF. Error: {}").format(e)
    N, D = self.X.shape
    # Max iterations
    self.n_iter = kwargs.get('n_iter', 500)
    # Convergence tolerance
    self.tol = kwargs.get('tol', 1e-6)
    # Initial weights
    self.w0 = kwargs.get('w0', np.zeros(D))
    # Sample for SGD?
    self.sample = kwargs.get('sample', True)
    self.n_samples = kwargs.get('n_samples', 100)
    # Elastic net mixing parameter
    self.alpha = kwargs.get('alpha', 0)
    if not (0 <= self.alpha <= 1):
      raise ValueError("Mixing parameter must be in [0,1]")
    # Learning rate and decay
    self.rate = kwargs.get('rate', 0.01)
    self.decay = kwargs.get('decay', 1)
    # Parameters to not regularize
    self.unreg = kwargs.get('unreg', [])
    # Use warm starts?
    self.warm_starts = kwargs.get('warm_starts', False)
    # Regularization
    if ('mu_seq' in kwargs) and (kwargs['mu_seq'] is not None):
      self.mu_seq = np.ravel(kwargs['mu_seq'])
      if not np.all(self.mu_seq >= 0):
        raise ValueError("Penalty parameters must be non-negative")
      self.mu_seq.sort()
    else:
      n_mu = kwargs.get('n_mu', 5)
      mu_min_ratio = kwargs.get('mu_min_ratio', 1e-6)
      self.mu_seq = get_mu_seq(n_mu, self.rate, self.alpha, mu_min_ratio)
    # Evidence/marginals
    self.evidence, self.marginals = None, None
    if 'evidence' in kwargs:
      self.evidence = np.ravel(kwargs['evidence'])
      if len(self.evidence) != N:
        raise ValueError("Need {} evidence values".format(N))
    if 'marginals' in kwargs:
      self.marginals = np.ravel(kwargs['marginals'])
      if not (np.all(self.marginals >= 0) and np.all(self.marginals <=1 ) and
              len(self.marginals) == N):
        raise ValueError("Need {} marginals in range [0,1]".format(N))
    if self.marginals is not None and self.evidence is not None:
      warnings.warn("Both evidence and marginals defined. Only using marginals.")
    # Be chatty?
    self.verbose = kwargs.get('verbose', False)
    
class pipeline_learning_opts:
  def __init__(self, kwargs):
    self.F, self.LF = kwargs.get('F', None), kwargs.get('LF', None)
    if self.F is None or self.LF is None:
      raise ValueError("Must provide X matrix, or F and LF matrices")
    if self.F.shape[0] != self.LF.shape[0]:
      raise ValueError("F and LF have different number of rows")
    if type(self.F) != np.ndarray and not sparse.issparse(self.F):
      raise TypeError("F should be np.ndarray type or scipy sparse.")
    if type(self.LF) != np.ndarray and not sparse.issparse(self.LF):
      raise TypeError("LF should be np.ndarray type or scipy sparse.")
    N, D, R = self.F.shape[0], self.F.shape[1], self.LF.shape[1]
    # Separate LF and feats arg sets
    self.lf_args, self.feats_args = dict(), dict()
    # Max iterations
    n_iter = kwargs.get('n_iter', 500)
    self.lf_args['n_iter'] = kwargs.get('n_iter_lf', n_iter)
    self.feats_args['n_iter'] = kwargs.get('n_iter_feats', n_iter)
    # Convergence tolerance
    tol = kwargs.get('tol', 1e-6)
    self.lf_args['tol'], self.feats_args['tol'] = tol, tol
    # Initial weights
    lf_mult = kwargs.get('w0_mult_lf', 1)
    self.lf_args['w0'] = lf_mult * kwargs.get('w0_lf', np.ones(R))
    self.feats_args['w0'] = kwargs.get('w0_feats', np.zeros(D))
    # Sample for SGD?
    sample = kwargs.get('sample', True)
    self.lf_args['sample'], self.feats_args['sample'] = sample, sample
    n_samp = kwargs.get('n_samples', 100)
    self.lf_args['n_samples'], self.feats_args['n_samples'] = n_samp, n_samp
    # Elastic net mixing parameter
    alpha = kwargs.get('alpha', 0)
    self.feats_args['alpha'] = kwargs.get('alpha_feats', alpha)
    if not (0 <= self.feats_args['alpha'] <= 1):
      raise ValueError("Mixing parameter must be in [0,1]")
    # Learning rate and decay
    rate = kwargs.get('rate', 0.01)
    decay = kwargs.get('decay', 1)
    self.lf_args['rate'] = kwargs.get('rate_lf', rate)
    self.lf_args['decay'] = kwargs.get('decay_lf', decay)
    self.feats_args['rate'] = kwargs.get('rate_feats', rate)
    self.feats_args['decay'] = kwargs.get('decay_feats', decay)
    # Parameters to not regularize
    unreg = kwargs.get('unreg', [])
    self.feats_args['unreg'] = kwargs.get('unreg_feats', unreg)
    # Use warm starts?
    warm_starts = kwargs.get('warm_starts', False)
    self.feats_args['warm_starts'] = kwargs.get('warm_starts_feats',
                                                 warm_starts)
    # Regularization
    self.lf_args['mu_seq'] = kwargs.get('mu_lf', 1e-7)
    if ('mu_seq' in kwargs) and ('mu_seq_feats' not in kwargs):
      kwargs['mu_seq_feats'] = kwargs['mu_seq']
    if ('mu_seq_feats' in kwargs) and (kwargs['mu_seq_feats'] is not None):
      self.feats_args['mu_seq'] = np.ravel(kwargs['mu_seq_feats'])
      if not np.all(self.feats_args['mu_seq'] >= 0):
        raise ValueError("Penalty parameters must be non-negative")
      self.feats_args['mu_seq'].sort()
    else:
      n_mu_feats = kwargs.get('n_mu_feats', 5)
      mu_min_ratio_feats = kwargs.get('mu_min_ratio_feats', 1e-6)
      self.feats_args['mu_seq'] = get_mu_seq(n_mu_feats,
                                             self.feats_args['rate'],
                                             self.feats_args['alpha'],
                                             mu_min_ratio_feats)
    # Evidence/marginals
    if 'evidence' in kwargs:
      self.lf_args['evidence'] = np.ravel(kwargs['evidence'])
      if len(self.lf_args['evidence']) != N:
        raise ValueError("Need {} evidence values".format(N))
    # Be chatty?
    verbose = kwargs.get('verbose', False)
    self.lf_args['verbose'], self.feats_args['verbose'] = verbose, verbose

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

def learn_elasticnet_logreg(**kwargs):
  """ Dispatcher for learning """
  if kwargs.get('pipeline', True):
    return pipeline_learn_elasticnet_logreg(**kwargs)
  else:
    return joint_learn_elasticnet_logreg(**kwargs)
    
def pipeline_learn_elasticnet_logreg(**kwargs):
  """ Pipeline learning """
  # Get options
  opts = pipeline_learning_opts(kwargs)
  # Learn LF weights  
  lfm = opts.lf_args['mu_seq']
  if opts.lf_args['verbose']:
    print "Learning LF accuracies with mu={}".format(lfm)
  lf_weights = joint_learn_elasticnet_logreg(X=opts.LF, **opts.lf_args)[lfm]
  marginals = odds_to_prob(np.ravel(opts.LF.dot(lf_weights)))
  # Learn feature weights
  if opts.feats_args['verbose']:
    print "Learning feature weights"
  feat_weights = joint_learn_elasticnet_logreg(X=opts.F, marginals=marginals,
                                               **opts.feats_args)
  return lf_weights, feat_weights, marginals

def joint_learn_elasticnet_logreg(**kwargs):
  """ Perform SGD wrt the weights w
       * w0 is the initial guess for w
       * sample and n_samples determine SGD batch size
       * alpha is the elastic net penalty mixing parameter (0=ridge, 1=lasso)
       * mu is the sequence of elastic net penalties to search over
  """
  # Get options
  opts = joint_learning_opts(kwargs)
  # Data matrix
  X = opts.X
  N, R = X.shape
  # Pre-generate other matrices
  Xt = X.transpose()
  Xt_abs = abs_sparse(Xt) if sparse.issparse(Xt) else np.abs(Xt)  
  # Initial weights
  w0 = opts.w0
  # Transform marginals  
  if opts.marginals is not None:
    t,f = opts.marginals, 1-opts.marginals
  # Training setup
  verbose = opts.verbose
  weights = dict()  
  # Search over penalty parameter values
  for mu in opts.mu_seq:
    if verbose:
      print "Begin training for mu = {}".format(mu)
    w = w0.copy()
    g = np.zeros(R)
    l = np.zeros(R)
    g_size = 0
    cur_rate = opts.rate
    # Gradient descent
    for step in range(opts.n_iter):
      # Get the expected LF accuracy
      if opts.marginals is None:
        t,f = sample_data(X, w, n_samples=opts.n_samples) if opts.sample\
              else exact_data(X, w, opts.evidence)
      p_correct, n_pred = transform_sample_stats(Xt, t, f, Xt_abs)
      # Get the "empirical log odds"; NB: this assumes one is correct, clamp is for sampling...
      l = np.clip(log_odds(p_correct), -10, 10)
      # SGD step with normalization by the number of samples
      g0 = (n_pred*(w - l)) / np.sum(n_pred)
      # Momentum term for faster training
      g = 0.95*g0 + 0.05*g
      # Check for convergence
      wn = np.linalg.norm(w, ord=2)
      g_size = np.linalg.norm(g, ord=2)
      if step % 100 == 0 and verbose:    
        print "\tLearning epoch = {}\tGradient mag. = {:.6f}".format(step, g_size) 
      if (wn < 1e-12 or g_size / wn < opts.tol) and step >= 10:
        if verbose:
          print "SGD converged for mu={:.3f} after {} steps".format(mu, step)
        break
      # Update weights
      cur_rate *= opts.decay
      w -= cur_rate * g
      # Store weights to not be regularized      
      w_unreg = w[opts.unreg].copy()
      # Apply elastic net penalty
      soft = np.abs(w) - cur_rate * opts.alpha * mu
      ridge_pen = (1 + (1-opts.alpha) * cur_rate * mu)
      #          \ell_1 penalty by soft thresholding        |  \ell_2 penalty
      w = (np.sign(w)*np.select([soft>0], [soft], default=0)) / ridge_pen
      # Unregularize
      w[opts.unreg] = w_unreg    
    # SGD did not converge    
    else:
      print "Final gradient magnitude for mu={:.3f}: {:.3f}".format(mu, g_size)
    # Store result and set warm start for next penalty
    weights[mu] = w.copy()
    if opts.warm_starts:
      w0 = w
  # Return learned weights  
  return weights
  
def get_mu_seq(n, rate, alpha, min_ratio):
  mv = (max(float(1 + rate * 10), float(rate * 11)) / (alpha + 1e-3))
  return np.logspace(np.log10(mv * min_ratio), np.log10(mv), n)
  