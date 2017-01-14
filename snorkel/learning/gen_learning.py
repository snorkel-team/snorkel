from .constants import *
from .disc_learning import NoiseAwareModel
from .utils import marginals_to_labels, MentionScorer, odds_to_prob
from numbskull import NumbSkull
from numbskull.inference import FACTORS
from numbskull.numbskulltypes import Weight, Variable, Factor, FactorToVar
import numpy as np
import random
import scipy.sparse as sparse
from utils import exact_data, log_odds, odds_to_prob, sample_data, sparse_abs, transform_sample_stats


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
            soft      = np.abs(w) - mu
            ridge_pen = (1 + (1-alpha) * mu)

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
    
    def save(self, session, version):
        raise NotImplementedError("Not implemented for generative model.")

    def load(self, session, version):
        raise NotImplementedError("Not implemented for generative model.")


class GenerativeModelWeights(object):

    def __init__(self, n):
        self.n = n
        self.class_prior = 0.0
        self.lf_accuracy_log_odds = np.zeros(n, dtype=np.float64)
        for optional_name in GenerativeModel.optional_names:
            setattr(self, optional_name, np.zeros(n, dtype=np.float64))

        for dep_name in GenerativeModel.dep_names:
            setattr(self, dep_name, sparse.lil_matrix((n, n), dtype=np.float64))

    def lf_accuracy(self):
	return 1.0 / (1.0 + np.exp(-self.lf_accuracy_log_odds)) 

    def is_sign_sparsistent(self, other, threshold=0.1):
        if self.n != other.n:
            raise ValueError("Dimension mismatch. %d versus %d" % (self.n, other.n))

        if not self._weight_is_sign_sparsitent(self.class_prior, other.class_prior, threshold):
            return False

        for i in range(self.n):
            if not self._weight_is_sign_sparsitent(
                    self.lf_accuracy_log_odds[i], other.lf_accuracy_log_odds[i], threshold):
                return False

        for name in GenerativeModel.optional_names:
            for i in range(self.n):
                if not self._weight_is_sign_sparsitent(
                        getattr(self, name)[i], getattr(other, name)[i], threshold):
                    return False

        for name in GenerativeModel.dep_names:
            for i in range(self.n):
                for j in range(self.n):
                    if not self._weight_is_sign_sparsitent(
                            getattr(self, name)[i, j], getattr(other, name)[i, j], threshold):
                        return False

        return True

    def _weight_is_sign_sparsitent(self, w1, w2, threshold):
        if abs(w1) <= threshold and abs(w2) <= threshold:
            return True
        elif w1 > threshold and w2 > threshold:
            return True
        elif w1 < -1 * threshold and w2 < -1 * threshold:
            return True
        else:
            return False


class GenerativeModel(object):
    """
    A generative model for data programming for binary classification.

    Supports dependencies among labeling functions.

    :param class_prior: whether to include class label prior factors
    :param lf_prior: whether to include labeling function prior factors
    :param lf_propensity: whether to include labeling function propensity factors
    :param lf_class_propensity: whether to include class-specific labeling function propensity factors
    :param seed: seed for initializing state of Numbskull variables
    """
    def __init__(self, class_prior=False, lf_prior=False, lf_propensity=False, lf_class_propensity=False, seed=271828):
        self.class_prior = class_prior
        self.lf_prior = lf_prior
        self.lf_propensity = lf_propensity
        self.lf_class_propensity = lf_class_propensity
        self.weights = None

        self.rng = random.Random()
        self.rng.seed(seed)

    # These names of factor types are for the convenience of several methods that perform the same operations over
    # multiple types, but this class's behavior is not fully specified here. Other methods, such as marginals(),
    # as well as maps defined within methods, require manual adjustments to implement changes.
    #
    # These names are also used by other related classes, such as GenerativeModelParameters
    optional_names = ('lf_prior', 'lf_propensity', 'lf_class_propensity')
    dep_names = ('dep_similar', 'dep_fixing', 'dep_reinforcing', 'dep_exclusive')

    def train(self, L, y=None, deps=(), init_acc = 1.0, epochs=100, step_size=None, decay=0.99, reg_param=0.1, reg_type=2, verbose=False,
              truncation=10, burn_in=50, timer=None):
	step_size = step_size or 1.0 / L.shape[0]
	reg_param_scaled = reg_param / L.shape[0]
        self._process_dependency_graph(L, deps)
        weight, variable, factor, ftv, domain_mask, n_edges = self._compile(L, y, init_acc)
        fg = NumbSkull(n_inference_epoch=0, n_learning_epoch=epochs, stepsize=step_size, decay=decay,
                       reg_param=reg_param_scaled, regularization=reg_type, truncation=truncation,
                       quiet=(not verbose), verbose=verbose, learn_non_evidence=True, burn_in=burn_in)
        fg.loadFactorGraph(weight, variable, factor, ftv, domain_mask, n_edges)
        if timer is not None:
            timer.start()
        fg.learning(out=False)
        if timer is not None:
            timer.end()
        self._process_learned_weights(L, fg)

    def marginals(self, L):
        if self.weights is None:
            raise ValueError("Must fit model with train() before computing marginal probabilities.")

        marginals = np.ndarray(L.shape[0], dtype=np.float64)

        for i in range(L.shape[0]):
            logp_true = self.weights.class_prior
            logp_false = -1 * self.weights.class_prior

            for _, j in zip(*L[i].nonzero()):
                if L[i, j] == 1:
                    logp_true  += self.weights.lf_accuracy_log_odds[j]
                    logp_false -= self.weights.lf_accuracy_log_odds[j]
                    logp_true  += self.weights.lf_class_propensity[j]
                    logp_false -= self.weights.lf_class_propensity[j]
                elif L[i, j] == -1:
                    logp_true  -= self.weights.lf_accuracy_log_odds[j]
                    logp_false += self.weights.lf_accuracy_log_odds[j]
                    logp_true  += self.weights.lf_class_propensity[j]
                    logp_false -= self.weights.lf_class_propensity[j]
                else:
                    ValueError("Illegal value at %d, %d: %d. Must be in {-1, 0, 1}." % (i, j, L[i, j]))

                for _, k in zip(*L[i].nonzero()):
                    if j != k:
                        if L[i, j] == -1 and L[i, k] == 1:
                            logp_true += self.weights.dep_fixing[j, k]
                        elif L[i, j] == 1 and L[i, k] == -1:
                            logp_false += self.weights.dep_fixing[j, k]

                        if L[i, j] == 1 and L[i, k] == 1:
                            logp_true += self.weights.dep_reinforcing[j, k]
                        elif L[i, j] == -1 and L[i, k] == -1:
                            logp_false += self.weights.dep_reinforcing[j, k]

            marginals[i] = 1 / (1 + np.exp(logp_false - logp_true))

        return marginals

    def score(self, session, X_test, test_labels, gold_candidate_set=None, b=0.5, set_unlabeled_as_neg=True,
              display=True, scorer=MentionScorer, **kwargs):
        
        # Get the test candidates
        test_candidates = [X_test.get_candidate(session, i) for i in xrange(X_test.shape[0])]

        # Initialize scorer
        s               = scorer(test_candidates, test_labels, gold_candidate_set)
        test_marginals  = self.marginals(X_test, **kwargs)

        return s.score(test_marginals, train_marginals=None, b=b,
                       set_unlabeled_as_neg=set_unlabeled_as_neg, display=display)

    def _process_dependency_graph(self, L, deps):
        """
        Processes an iterable of triples that specify labeling function dependencies.

        The first two elements of the triple are the labeling functions to be modeled as dependent. The labeling
        functions are specified using their column indices in `L`. The third element is the type of dependency.
        Options are :const:`DEP_SIMILAR`, :const:`DEP_FIXING`, :const:`DEP_REINFORCING`, and :const:`DEP_EXCLUSIVE`.

        The results are :class:`scipy.sparse.csr_matrix` objects that represent directed adjacency matrices. They are
        set as various GenerativeModel members, two for each type of dependency, e.g., `dep_similar` and `dep_similar_T`
        (its transpose for efficient inverse lookups).

        :param deps: iterable of tuples of the form (lf_1, lf_2, type)
        """
        dep_name_map = {
            DEP_SIMILAR: 'dep_similar',
            DEP_FIXING: 'dep_fixing',
            DEP_REINFORCING: 'dep_reinforcing',
            DEP_EXCLUSIVE: 'dep_exclusive'
        }

        for dep_name in GenerativeModel.dep_names:
            setattr(self, dep_name, sparse.lil_matrix((L.shape[1], L.shape[1])))

        for lf1, lf2, dep_type in deps:
            if lf1 == lf2:
                raise ValueError("Invalid dependency. Labeling function cannot depend on itself.")

            if dep_type in dep_name_map:
                dep_mat = getattr(self, dep_name_map[dep_type])
            else:
                raise ValueError("Unrecognized dependency type: " + unicode(dep_type))

            dep_mat[lf1, lf2] = 1

        for dep_name in GenerativeModel.dep_names:
            setattr(self, dep_name, getattr(self, dep_name).tocoo(copy=True))

    def _compile(self, L, y, init_acc):
        """
        Compiles a generative model based on L and the current labeling function dependencies.
        """
        m, n = L.shape

        n_weights = 1 if self.class_prior else 0

        n_weights += n
        for optional_name in GenerativeModel.optional_names:
            if getattr(self, optional_name):
                n_weights += n
        for dep_name in GenerativeModel.dep_names:
            n_weights += getattr(self, dep_name).getnnz()

        n_vars = m * (n + 1)
        n_factors = m * n_weights

        n_edges = 1 if self.class_prior else 0
        n_edges += 2 * n
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
        if self.class_prior:
            weight[0]['isFixed'] = False
            weight[0]['initialValue'] = np.float64(0)
            w_off = 1
        else:
            w_off = 0

        for i in range(w_off, w_off + n):
            weight[i]['isFixed'] = False
            weight[i]['initialValue'] = np.float64(init_acc + .1 - .2 * self.rng.random())

        w_off += n
        for i in range(w_off, weight.shape[0]):
            weight[i]['isFixed'] = False
            weight[i]['initialValue'] = np.float64(0.0)


        #
        # Compiles variable matrix
        #
        for i in range(m):
            variable[i]['isEvidence'] = False if (y is None or i not in y) else True
            variable[i]['initialValue'] = self.rng.randrange(0, 2) if (y is None or i not in y) else 1 if y[i] == 1 else 0
            variable[i]["dataType"] = 0
            variable[i]["cardinality"] = 2


        for i in range(m):
            for j in range(n):
                index = m + n * i + j
                variable[index]["isEvidence"] = 1
                if L[i, j] == 1:
                    variable[index]["initialValue"] = 2
                elif L[i, j] == 0:
                    variable[index]["initialValue"] = 1
                elif L[i, j] == -1:
                    variable[index]["initialValue"] = 0
                else:
                    raise ValueError("Invalid labeling function output in cell (%d, %d): %d. "
                                     "Valid values are 1, 0, and -1. " % i, j, L[i, j])
                variable[index]["dataType"] = 0
                variable[index]["cardinality"] = 3


        #
        # Compiles factor and ftv matrices
        #

        # Class prior
        if self.class_prior:
            for i in range(m):
                factor[i]["factorFunction"] = FACTORS["DP_GEN_CLASS_PRIOR"]
                factor[i]["weightId"] = 0
                factor[i]["featureValue"] = 1
                factor[i]["arity"] = 1
                factor[i]["ftv_offset"] = i

                ftv[i]["vid"] = i

            f_off = m
            ftv_off = m
            w_off = 1
        else:
            f_off = 0
            ftv_off = 0
            w_off = 0


        # Factors over labeling function outputs
        f_off, ftv_off, w_off = self._compile_output_factors(L, factor, f_off, ftv, ftv_off, w_off, "DP_GEN_LF_ACCURACY",
                                                             (lambda m, n, i, j: i, lambda m, n, i, j: m + n * i + j))

        optional_name_map = {
            'lf_prior':
                ('DP_GEN_LF_PRIOR', (
                    lambda m, n, i, j: m + n * i + j,)),
            'lf_propensity':
                ('DP_GEN_LF_PROPENSITY', (
                    lambda m, n, i, j: m + n * i + j,)),
            'lf_class_propensity':
                ('DP_GEN_LF_CLASS_PROPENSITY', (
                    lambda m, n, i, j: i,
                    lambda m, n, i, j: m + n * i + j)),
        }

        for optional_name in GenerativeModel.optional_names:
            if getattr(self, optional_name):
                f_off, ftv_off, w_off = self._compile_output_factors(L, factor, f_off, ftv, ftv_off, w_off,
                                                                     optional_name_map[optional_name][0],
                                                                     optional_name_map[optional_name][1])


        # Factors for labeling function dependencies
        dep_name_map = {
            'dep_similar':
                ('EQUAL', (
                    lambda m, n, i, j, k: m + n * i + j,
                    lambda m, n, i, j, k: m + n * i + k)),
            'dep_fixing':
                ('DP_GEN_DEP_FIXING', (
                    lambda m, n, i, j, k: i,
                    lambda m, n, i, j, k: m + n * i + j,
                    lambda m, n, i, j, k: m + n * i + k)),
            'dep_reinforcing':
                ('DP_GEN_DEP_REINFORCING', (
                    lambda m, n, i, j, k: i,
                    lambda m, n, i, j, k: m + n * i + j,
                    lambda m, n, i, j, k: m + n * i + k)),
            'dep_exclusive':
                ('DP_GEN_DEP_EXCLUSIVE', (
                    lambda m, n, i, j, k: m + n * i + j,
                    lambda m, n, i, j, k: m + n * i + k))
        }

        for dep_name in GenerativeModel.dep_names:
            mat = getattr(self, dep_name)
            for i in range(len(mat.data)):
                f_off, ftv_off, w_off = self._compile_dep_factors(L, factor, f_off, ftv, ftv_off, w_off,
                                                                  mat.row[i], mat.col[i],
                                                                  dep_name_map[dep_name][0],
                                                                  dep_name_map[dep_name][1])


        return weight, variable, factor, ftv, domain_mask, n_edges

    def _compile_output_factors(self, L, factors, factors_offset, ftv, ftv_offset, weight_offset, factor_name, vid_funcs):
        """
        Compiles factors over the outputs of labeling functions, i.e., for which there is one weight per labeling
        function and one factor per labeling function-candidate pair.
        """
        m, n = L.shape

        for i in range(m):
            for j in range(n):
                factors_index = factors_offset + n * i + j
                ftv_index = ftv_offset + len(vid_funcs) * (n * i + j)

                factors[factors_index]["factorFunction"] = FACTORS[factor_name]
                factors[factors_index]["weightId"] = weight_offset + j
                factors[factors_index]["featureValue"] = 1
                factors[factors_index]["arity"] = len(vid_funcs)
                factors[factors_index]["ftv_offset"] = ftv_index

                for i_var, vid_func in enumerate(vid_funcs):
                    ftv[ftv_index + i_var]["vid"] = vid_func(m, n, i, j)

        return factors_offset + m * n, ftv_offset + len(vid_funcs) * m * n, weight_offset + n

    def _compile_dep_factors(self, L, factors, factors_offset, ftv, ftv_offset, weight_offset, j, k, factor_name, vid_funcs):
        """
        Compiles factors for dependencies between pairs of labeling functions (possibly also depending on the latent
        class label).
        """
        m, n = L.shape

        for i in range(m):
            factors_index = factors_offset + i
            ftv_index = ftv_offset + len(vid_funcs) * i

            factors[factors_index]["factorFunction"] = FACTORS[factor_name]
            factors[factors_index]["weightId"] = weight_offset
            factors[factors_index]["featureValue"] = 1
            factors[factors_index]["arity"] = len(vid_funcs)
            factors[factors_index]["ftv_offset"] = ftv_index

            for i_var, vid_func in enumerate(vid_funcs):
                ftv[ftv_index + i_var]["vid"] = vid_func(m, n, i, j, k)

        return factors_offset + m, ftv_offset + len(vid_funcs) * m, weight_offset + 1

    def _process_learned_weights(self, L, fg):
        _, n = L.shape

        w = fg.getFactorGraph().getWeights()
        weights = GenerativeModelWeights(n)

        if self.class_prior:
            weights.class_prior = w[0]
            w_off = 1
        else:
            w_off = 0

        weights.lf_accuracy_log_odds = np.copy(w[w_off:w_off + n])
        w_off += n

        for optional_name in GenerativeModel.optional_names:
            if getattr(self, optional_name):
                setattr(weights, optional_name, np.copy(w[w_off:w_off + n]))
                w_off += n

        for dep_name in self.dep_names:
            mat = getattr(self, dep_name)
            weight_mat = sparse.lil_matrix((n, n))

            for i in range(len(mat.data)):
                if w[w_off] != 0:
                    weight_mat[mat.row[i], mat.col[i]] = w[w_off]
                w_off += 1

            setattr(weights, dep_name, weight_mat.tocsr(copy=True))

        self.weights = weights
