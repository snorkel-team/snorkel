from .disc_learning import NoiseAwareModel
from ..models import Parameter, ParameterSet
from numbskull.inference import FACTORS
from numbskull.factorgraph import FactorGraph
from numbskull.numbskulltypes import Weight, Variable, Factor, FactorToVar
import numpy as np
import scipy.sparse as sparse
from utils import exact_data, log_odds, odds_to_prob, sample_data, sparse_abs, transform_sample_stats

DEP_SIMILAR = 0
DEP_FIXING = 1
DEP_REINFORCING = 2
DEP_EXCLUSIVE = 3


class NaiveBayes(NoiseAwareModel):
    def __init__(self, bias_term=False):
        self.w         = None
        self.bias_term = bias_term

    def train(self, X, n_iter=1000, w0=None, rate=DEFAULT_RATE, alpha=DEFAULT_ALPHA, mu=DEFAULT_MU,
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

    def train(self, L, deps=()):
        self.L = L
        self._process_dependency_graph(deps)
        self._compile()

    def _process_dependency_graph(self, deps):
        """
        Processes an iterable of triples that specify labeling function dependencies.

        The first two elements of the triple are the labeling functions to be modeled as dependent. The labeling
        functions are specified using their column indices in `self.L`. The third element is the type of dependency.
        Options are :const:`DEP_SIMILAR`, :const:`DEP_FIXING`, :const:`DEP_REINFORCING`, and :const:`DEP_EXCLUSIVE`.

        The results are :class:`scipy.sparse.csr_matrix` objects that represent directed adjacency matrices. They are
        set as various GenerativeModel members, two for each type of dependency, e.g., `dep_similar` and `dep_similar_T`
        (its transpose for efficient inverse lookups).

        :param deps: iterable of tuples of the form (lf_1, lf_2, type)
        """
        self.dep_similar = sparse.lil_matrix(self.L.shape[1], self.L.shape[1])
        self.dep_fixing = sparse.lil_matrix(self.L.shape[1], self.L.shape[1])
        self.dep_reinforcing = sparse.lil_matrix(self.L.shape[1], self.L.shape[1])
        self.dep_exclusive = sparse.lil_matrix(self.L.shape[1], self.L.shape[1])

        for lf1, lf2, dep_type in deps:
            if lf1 == lf2:
                raise ValueError("Invalid dependency. Labeling function cannot depend on itself.")

            if dep_type == DEP_SIMILAR:
                dep_mat = self.dep_similar
            elif dep_type == DEP_FIXING:
                dep_mat = self.dep_fixing
            elif dep_type == DEP_REINFORCING:
                dep_mat = self.dep_reinforcing
            elif dep_type == DEP_EXCLUSIVE:
                dep_mat = self.dep_exclusive
            else:
                raise ValueError("Unrecognized dependency type: " + unicode(dep_type))

            dep_mat[lf1, lf2] = 1

        for dep_name in ('similar', 'fixing', 'reinforcing', 'exclusive'):
            dep_name = 'dep_' + dep_name
            setattr(self, dep_name, getattr(self, dep_name).tocsr(copy=True))
            setattr(self, dep_name + '_T', getattr(self, dep_name).transpose().tocsr(copy=True))

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
        n_weights += self.dep_similar.getnnz() + self.dep_fixing.getnnz() + \
                     self.dep_reinforcing.getnnz() + self.dep_exclusive.getnnz()

        n_vars = m * (n + 1)
        n_factors = m * n_weights

        n_edges = 1 + 2 * n
        if self.lf_prior:
            n_edges += n
        if self.lf_propensity:
            n_edges += n
        if self.lf_class_propensity:
            n_edges += 2 * n
        n_edges += 2 * self.dep_similar.getnnz() + 3 * self.dep_fixing.getnnz() + \
                   3 * self.dep_reinforcing.getnnz() + 2 * self.dep_exclusive.getnnz()
        n_edges *= m

        weight = np.zeros(n_weights, Weight)
        variable = np.zeros(n_vars, Variable)
        factor = np.zeros(n_factors, Factor)
        ftv = np.zeros(n_edges, FactorToVar)
        domain_mask = np.zeros(n_vars, np.bool)

        #
        # Compiles weight matrix
        #
        weight[0]['isFixed'] = False
        # In most information extraction tasks, the label distribution is weighted towards the negative class, so we
        # initialize accordingly
        weight[0]['initialValue'] = -1
        for i in range(1, weight.shape[0]):
            weight[i]['isFixed'] = False
            weight[i]['initialValue'] = 1.1 - .2 * random.random()

        #
        # Compiles variable matrix
        #
        init_values = (0, 1)
        for i in range(m):
            variable[i]['isEvidence'] = False
            variable[i]['initialValue'] = random.choose(init_values)
            variable[i]["dataType"] = 1
            variable[i]["cardinality"] = 2

        for i in range(m):
            for j in range(n):
                index = m + n * i + j
                variable[index]["isEvidence"] = 1
                if self.L[i, j] == 1:
                    variable[index]["initialValue"] = 2
                elif self.L[i, j] == 0:
                    variable[index]["initialValue"] = 1
                elif self.L[i, j] == -1:
                    variable[index]["initialValue"] = 0
                else:
                    raise ValueError("Invalid labeling function output in cell (%d, %d): %d. "
                                     "Valid values are 1, 0, and -1. " % i, j, self.L[i, j])
                variable[index]["dataType"] = 1
                variable[index]["cardinality"] = 3

        #
        # Compiles factor and ftv matrices
        #

        # Class prior
        for i in range(m):
            factor[i]["factorFunction"] = FACTORS["FUNC_DP_GEN_CLASS_PRIOR"]
            factor[i]["weightId"] = 0
            factor[i]["featureValue"] = 1
            factor[i]["arity"] = 1
            factor[i]["ftv_offset"] = i

            ftv[i]["vid"] = i

        # Factors over labeling function outputs
        f_off, ftv_off, w_off = self._compile_output_factors(factor, m, ftv, m, 1,
                                                             "FUNC_DP_GEN_LF_ACCURACY",
                                                             (lambda m, n, i, j: i,
                                                              lambda m, n, i, j: m + n * i + j))
        if self.lf_prior:
            f_off, ftv_off, w_off = self._compile_output_factors(factor, f_off, ftv, ftv_off, w_off,
                                                                 "FUNC_DP_GEN_LF_PRIOR",
                                                                 (lambda m, n, i, j: m + n * i + j))
        if self.lf_propensity:
            f_off, ftv_off, w_off = self._compile_output_factors(factor, f_off, ftv, ftv_off, w_off,
                                                                 "FUNC_DP_GEN_LF_PROPENSITY",
                                                                 (lambda m, n, i, j: m + n * i + j))

        if self.lf_class_propensity:
            f_off, ftv_off, w_off = self._compile_output_factors(factor, f_off, ftv, ftv_off, w_off,
                                                                 "FUNC_DP_GEN_LF_CLASS_PROPENSITY",
                                                                 (lambda m, n, i, j: m + n * i + j))

    def _compile_output_factors(self, factors, factors_offset, ftv, ftv_offset, weight_offset, factor_name, vid_funcs):
        """
        Compiles factors over the outputs of labeling functions, i.e., for which there is one weight per labeling
        function and one factor per labeling function-candidate pair.
        """
        m, n = self.L.shape

        for i in range(m):
            for j in range(n):
                factors_index = factors_offset + n * i + j
                ftv_index = ftv_offset + len(vid_funcs) * (n * i + j)

                factors[factors_index]["factorFunction"] = FACTORS[factor_name]
                factors[factors_index]["weightId"] = weight_offset + j
                factors[factors_index]["featureValue"] = 1
                factors[factors_index]["arity"] = len(vid_funcs)
                factors[factors_index]["ftv_offset"] = ftv_offset

                for i, vid_func in enumerate(vid_funcs):
                    ftv[ftv_index + i]["vid"] = vid_func(m, n, i, j)

        return factors_offset + m * n, ftv_offset + len(vid_funcs) * m * n, weight_offset + n
