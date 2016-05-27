import numpy as np
import scipy.sparse as sparse
import warnings

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

def learn_elasticnet_logreg(X, n_iter=500, tol=1e-6, w0=None, sample=True,
                            n_samples=100, alpha=0, mu_seq=None, n_mu=10,
                            mu_min_ratio=1e-6, rate=0.01, decay=1, 
                            warm_starts=False, evidence=None, marginals=None,
                            unreg=[], verbose=False):
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
    if len(evidence) != N:
      raise ValueError("Need {} evidence values".format(N))
  
  if marginals is not None:
    marginals = np.ravel(marginals)
    if not (np.all(marginals >= 0) and np.all(marginals <=1) and
            len(marginals) == N):
      raise ValueError("Need {} marginals in range [0,1]".format(N))
    t,f = marginals, 1-marginals

  if marginals is not None and evidence is not None:
    warnings.warn("Both evidence and marginals defined. Only using marginals.")

  weights = dict()
  
  # Search over penalty parameter values
  for mu in mu_seq:
    if verbose:
      print "Begin training for mu = {}".format(mu)
    w = w0.copy()
    g = np.zeros(R)
    l = np.zeros(R)
    g_size = 0
    cur_rate = rate
    # Take SGD steps
    for step in range(n_iter):
      # Get the expected LF accuracy
      if marginals is None:
        t,f = sample_data(X, w, n_samples=n_samples) if sample\
              else exact_data(X, w, evidence)
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
      if wn < 1e-12 or g_size / wn < tol:
        if verbose:
          print "SGD converged for mu={:.3f} after {} steps".format(mu, step)
        break

      # Update weights
      cur_rate *= decay
      w -= cur_rate * g
      
      w_unreg = w[unreg].copy()
      # Apply elastic net penalty
      soft = np.abs(w) - cur_rate * alpha * mu
      #          \ell_1 penalty by soft thresholding        |  \ell_2 penalty
      w = (np.sign(w)*np.select([soft>0], [soft], default=0)) / (1 + (1-alpha) * cur_rate * mu)
      w[unreg] = w_unreg
    
    # SGD did not converge    
    else:
      print "Final gradient magnitude for mu={:.3f}: {:.3f}".format(mu, g_size)

    # Store result and set warm start for next penalty
    weights[mu] = w.copy()
    if warm_starts:
      w0 = w
    
  return weights
  
def get_mu_seq(n, rate, alpha, min_ratio):
  mv = (max(float(1 + rate * 10), float(rate * 11)) / (alpha + 1e-3))
  return np.logspace(np.log10(mv * min_ratio), np.log10(mv), n)