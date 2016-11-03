import numpy as np
import scipy.sparse as sparse
import warnings

DEFAULT_MU = 1e-6
DEFAULT_RATE = 0.01
DEFAULT_ALPHA = 0.5

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


def exact_marginals_single_candidate(X, w):
    """
    This function computes the marginal probabilities of each class (of D classes) for **a single candidate**
    with true class Y; the output is a D-dim vector of these probabilities.

    Here, X is a M x D matrix, where M is the number of LFs, and D is the number of possible; 
    the ith row of X corresponds to the *distribution* of the LF's "vote" across possible values.
    This is just the softmax:

    P(Y=k | X; w) = exp(w^T X[:, k]) / \sum_l ( exp(w^T X[:, l]) )
    
    Our default convention is that if, for example, the ith LF votes only negatively on a value j
    of this candidate, then this would be expressed as having uniform distribution over all cols. except j.
    in row i of X.
    """
    z = np.exp(np.dot(w.T, X))
    return z / z.sum()


def compute_lf_accs(Xs, w):
    """
    This function computes the expected accuracies of each of the M LFs, outputting a M-dim vector.

    Here, Xs is a **list** of matrices X, as defined in exact_marginals_single_candidate;
    note that each X has the same number of rows (M), but varying number of columns.
    
    E[ accuracy of LF i ] =
    """
    M      = Xs[0].shape[0]
    accs   = np.zeros((M,1))
    n_pred = np.zeros(M)

    w = w.reshape(-1,1)

    # Iterate over the different LF distribution matrices for each candidate
    for i, X in enumerate(Xs):
        
        # Get the predicted class distribution for the candidate
        z = exact_marginals_single_candidate(X, w)

        # Get the expected accuracy of the LFs for this candidate
        # TODO: Check this...
        accs += np.dot(X,z.T) / np.linalg.norm(z) # M X D * D

        # Add whether there was a prediction made or not
        # TODO: Check this...
        n_pred += X.sum(1) #summing across rows 0/1

    p_correct = (1. / (n_pred + 1e-8)).reshape(-1,1) * accs
    return p_correct, n_pred


class NoiseAwareModel(object):
    """Simple abstract base class for a model."""
    def __init__(self):
        pass

    def train(self, X, training_marginals=None, **hyperparams):
        raise NotImplementedError()

    def marginals(self, X):
        raise NotImplementedError()

    def predict(self, X, thresh=0.5):
        """Return numpy array of elements in {-1,0,1} based on predicted marginal probabilities."""
        return np.array([1 if p > thresh else -1 if p < thresh else 0 for p in self.marginals(X)])


class LogReg(NoiseAwareModel):
    """Logistic regression."""
    def __init__(self, bias_term=False):
        self.w         = None
        self.bias_term = bias_term

    def train(self, Xs, n_iter=1000, w0=None, rate=DEFAULT_RATE, alpha=DEFAULT_ALPHA, \
            mu=DEFAULT_MU, tol=1e-6, verbose=True):
        """
        Xs is defined as in compute_lf_accs.

        Perform SGD wrt the weights w
        * n_iter:      Number of steps of SGD
        * w0:          Initial value for weights w
        * rate:        I.e. the SGD step size
        * alpha:       Elastic net penalty mixing parameter (0=ridge, 1=lasso)
        * mu:          Elastic net penalty
        * tol:         For testing for SGD convergence, i.e. stopping threshold
        """
        # Set up stuff
        N  = len(Xs)
        M  = Xs[0].shape[0]
        w0 = w0 if w0 is not None else np.zeros(M)

        # Initialize training
        w = w0.copy()
        g = np.zeros(M)
        l = np.zeros(M)
        g_size = 0

        # Gradient descent
        if verbose:
            print "Begin training for rate={}, mu={}".format(rate, mu)
        for step in range(n_iter):

            # Get the expected LF accuracies
            p_correct, n_pred = compute_lf_accs(Xs, w)

            # Get the "empirical log odds"; NB: this assumes one is correct, clamp is for sampling...
            l = np.clip(log_odds(p_correct), -10, 10).flatten()

            # SGD step with normalization by the number of samples
            g0 = (n_pred * (w - l)) / np.sum(n_pred)

            # Momentum term for faster training
            g = 0.95*g0 + 0.05*g

            # Check for convergence
            wn     = np.linalg.norm(w, ord=2)
            g_size = np.linalg.norm(g, ord=2)
            if step % 100 == 0 and verbose:
                print "\tLearning epoch = {}\tGradient mag. = {:.6f}".format(step, g_size)
            if (wn < 1e-12 or g_size / wn < tol) and step >= 10:
                if verbose:
                    print "SGD converged for mu={} after {} steps".format(mu, step)
                break

            #print "w", w.shape
            #print "g", g.shape
            #print rate

            #print w
            #print g

            #print "!!!!!",rate * g

            # Update weights
            w -= rate * g


            # Apply elastic net penalty
            w_bias    = w[-1]
            soft      = np.abs(w) - rate * alpha * mu
            ridge_pen = (1 + (1-alpha) * rate * mu)

            #          \ell_1 penalty by soft thresholding        |  \ell_2 penalty
            w = (np.sign(w)*np.select([soft>0], [soft], default=0)) / ridge_pen

            # Don't regularize the bias term
            if self.bias_term:
                w[-1] = w_bias

        # SGD did not converge
        else:
            if verbose:
                print "Final gradient magnitude for rate={}, mu={}: {:.3f}".format(rate, mu, g_size)

        # Return learned weights
        self.w = w

    def marginals(self, Xs):
        return [exact_marginals_single_candidate(X, self.w) for X in Xs]
