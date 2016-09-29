from .disc_learning import NoiseAwareModel
from ..models import Parameter, ParameterSet
from numbskull.inference import FACTORS
from numbskull.factorgraph import FactorGraph
from numbskull.numbskulltypes import Weight, Variable, Factor, FactorToVar
import numpy as np
import scipy.sparse as sparse
from utils import exact_data, sample_data, sparse_abs, transform_sample_stats

DEFAULT_MU = 1e-6
DEFAULT_RATE = 0.01
DEFAULT_ALPHA = 0.5

DEP_SIMILAR = 0
DEP_FIXING = 1
DEP_REINFORCING = 2
DEP_EXCLUSIVE = 3


class NaiveBayes(NoiseAwareModel):
    def __init__(self, bias_term=False):
        self.w         = None
        self.bias_term = bias_term

    def train(self, X, n_iter=1000, w0=None, rate=DEFAULT_RATE, alpha=DEFAULT_ALPHA, mu=DEFAULT_MU, \
            sample=False, n_samples=100, evidence=None, warm_starts=False, tol=1e-6, verbose=True):
        """
        Perform SGD wrt the weights w
        * n_iter:      Number of steps of SGD
        * w0:          Initial value for weights w
        * rate:        I.e. the SGD step size
        * alpha:       Elastic net penalty mixing parameter (0=ridge, 1=lasso)
        * mu:          Elastic net penalty
        * sample:      Whether to sample or not
        * n_samples:   Number of samples per SGD step
        * evidence:    Ground truth to condition on
        * warm_starts:
        * tol:         For testing for SGD convergence, i.e. stopping threshold
        """
        self.X_train = X

        # Set up stuff
        N, M   = X.shape
        print "="*80
        print "Training marginals (!= 0.5):\t%s" % N
        print "Features:\t\t\t%s" % M
        print "="*80
        Xt     = X.transpose()
        Xt_abs = sparse_abs(Xt) if sparse.issparse(Xt) else np.abs(Xt)
        w0     = w0 if w0 is not None else np.ones(M)

        # Initialize training
        w = w0.copy()
        g = np.zeros(M)
        l = np.zeros(M)
        g_size = 0

        # Gradient descent
        if verbose:
            print "Begin training for rate={}, mu={}".format(rate, mu)
        for step in range(n_iter):

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
            wn     = np.linalg.norm(w, ord=2)
            g_size = np.linalg.norm(g, ord=2)
            if step % 250 == 0 and verbose:
                print "\tLearning epoch = {}\tGradient mag. = {:.6f}".format(step, g_size)
            if (wn < 1e-12 or g_size / wn < tol) and step >= 10:
                if verbose:
                    print "SGD converged for mu={} after {} steps".format(mu, step)
                break

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

    def marginals(self, X):
        return odds_to_prob(X.dot(self.w))


class GenerativeModel(object):
    """

    :param lf_prior:
    :param lf_propensity:
    :param lf_class_propensity:
    """
    def __init__(self, lf_prior=True, lf_propensity=True, lf_class_propensity=True, seed=271828):
        self.L = None
        self.fg = None
        self.lf_prior = lf_prior
        self.lf_propensity = lf_propensity
        self.lf_class_propensity = lf_class_propensity
        self.seed = seed

    def train(self, L, deps):
        self.L = L
        self._process_dependency_graph(deps)
        self._compile()

    def _process_dependency_graph(self, deps):
        """
        Processes an iterable of triples that specify labeling function dependencies.

        The first two elements of the triple are the labeling functions to be modeled as dependent. The labeling
        functions are specified using their column indices in `self.L`. The third element is the type of dependency.
        Options are :const:`DEP_SIMILAR`, :const:`DEP_FIXING`, :const:`DEP_REINFORCING`, and :const:`DEP_EXCLUSIVE`.

        The results are lists of unique pairs of Label AnnotationKeys. They are set as various object members, one
        for each type of dependency.

        :param deps: iterable of tuples of the form (lf_1, lf_2, type)
        """
        #TODO: Redo as sparse matrices
        self.dep_similar = sparse.lil_matrix(self.L.shape[1], self.L.shape[1])
        self.dep_fixing = sparse.lil_matrix(self.L.shape[1], self.L.shape[1])
        self.dep_reinforcing = sparse.lil_matrix(self.L.shape[1], self.L.shape[1])
        self.dep_exclusive = sparse.lil_matrix(self.L.shape[1], self.L.shape[1])

        for lf1, lf2, dep_type in deps:
            if dep_type == DEP_SIMILAR:
                dep_set = self.dep_similar
            elif dep_type == DEP_FIXING:
                dep_set = self.dep_fixing
            elif dep_type == DEP_REINFORCING:
                dep_set = self.dep_reinforcing
            elif dep_type == DEP_EXCLUSIVE:
                dep_set = self.dep_exclusive
            else:
                raise ValueError("Unrecognized dependency type: " + unicode(dep_type))

            dep_set.add((lf1, lf2))

        self.dep_similar = sorted([dep for dep in self.dep_similar], key=lambda lf1, lf2: lf1 - 1 / float(lf2))
        self.dep_fixing = sorted([dep for dep in self.dep_fixing], key=lambda lf1, lf2: lf1 - 1 / float(lf2))
        self.dep_reinforcing = sorted([dep for dep in self.dep_reinforcing], key=lambda lf1, lf2: lf1 - 1 / float(lf2))
        self.dep_exclusive = sorted([dep for dep in self.dep_exclusive], key=lambda lf1, lf2: lf1 - 1 / float(lf2))

    def _compile(self):
        """
        Compiles a :class:`numbskull.factorgraph.FactorGraph` based on the current `self.L` and labeling function
        dependencies.

        The result is set as `self.fg`.
        """
        m, n = self.L.shape()

        n_weights = 1 + n
        if self.lf_prior:
            n_weights += n
        if self.lf_propensity:
            n_weights += n
        if self.lf_class_propensity:
            n_weights += n
        n_weights += len(self.dep_similar) + len(self.dep_fixing) + len(self.dep_reinforcing) + len(self.dep_exclusive)

        n_vars = m * (n+1)

        n_factors = m * (n+1)
        if self.lf_prior:
            n_factors += m * n
        if self.lf_propensity:
            n_factors += m * n
        if self.lf_class_propensity:
            n_factors += m * n
        n_factors += m * (len(self.dep_similar) + len(self.dep_fixing) +
                          len(self.dep_reinforcing) + len(self.dep_exclusive))

        n_edges = 0

        weight = np.zeros(n_weights, Weight)
        variable = np.zeros(n_vars, Variable)
        factor = np.zeros(n_factors, Factor)
        fmap = np.zeros(n_edges, FactorToVar)
        domain_mask = np.zeros(n_vars, np.bool)

        #
        #
        #

        variable[0]["isEvidence"] = 0
        variable[0]["initialValue"] = 0
        variable[0]["dataType"] = 0
        variable[0]["cardinality"] = 2

        variable[1]["isEvidence"] = 0
        variable[1]["initialValue"] = 0
        variable[1]["dataType"] = 0
        variable[1]["cardinality"] = 2

        factor[0]["factorFunction"] = value
        factor[0]["weightId"] = 0
        factor[0]["featureValue"] = 1
        factor[0]["arity"] = 2
        factor[0]["ftv_offset"] = 0

        fmap[0]["vid"] = 0
        fmap[1]["vid"] = 1
