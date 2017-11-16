from .classifier import Classifier
from numba import jit
import numbskull
from numbskull import NumbSkull
from numbskull.inference import FACTORS
from numbskull.numbskulltypes import Weight, Variable, Factor, FactorToVar
import numpy as np
import random
import scipy.sparse as sparse
from copy import copy
from pandas import DataFrame
from distutils.version import StrictVersion
from six.moves.cPickle import dump, load
import os

DEP_SIMILAR = 0
DEP_FIXING = 1
DEP_REINFORCING = 2
DEP_EXCLUSIVE = 3


class GenerativeModel(Classifier):
    """
    A generative model for data programming for binary classification.

    Supports dependencies among labeling functions.

    :param class_prior: whether to include class label prior factors
    :param lf_prior: whether to include labeling function prior factors
    :param lf_propensity: whether to include labeling function propensity
        factors
    :param lf_class_propensity: whether to include class-specific labeling
        function propensity factors
    :param seed: seed for initializing state of Numbskull variables
    """
    def __init__(self, class_prior=False, lf_prior=False, lf_propensity=False,
        lf_class_propensity=False, seed=271828, name=None):
        self.name = name or self.__class__.__name__
        try:
            numbskull_version = numbskull.__version__
        except:
            numbskull_version = "0.0"
        numbskull_require = "0.1"

        if StrictVersion(numbskull_version) < StrictVersion(numbskull_require):
            raise ValueError(
                "Snorkel requires Numbskull version %s, but version %s is installed." % (numbskull_require, numbskull_version))

        self.class_prior = class_prior
        self.lf_prior = lf_prior
        self.lf_propensity = lf_propensity
        self.lf_class_propensity = lf_class_propensity
        self.weights = None

        self.rng = np.random.RandomState()
        self.rng.seed(seed)
        set_numba_seeds(seed)

    # These names of factor types are for the convenience of several methods
    # that perform the same operations over multiple types, but this class's
    # behavior is not fully specified here. Other methods, such as marginals(),
    # as well as maps defined within methods, require manual adjustments to
    # implement changes.
    #
    # These names are also used by other related classes, such as
    # GenerativeModelParameters
    optional_names = ('lf_prior', 'lf_propensity', 'lf_class_propensity')
    dep_names = (
        'dep_similar', 'dep_fixing', 'dep_reinforcing', 'dep_exclusive'
    )

    def train(self, L, deps=(), LF_acc_prior_weights=None,
        LF_acc_prior_weight_default=1, labels=None, label_prior_weight=5,
        init_deps=0.0, init_class_prior=-1.0, epochs=30, step_size=None, 
        decay=1.0, reg_param=0.1, reg_type=2, verbose=False, truncation=10, 
        burn_in=5, cardinality=None, timer=None, candidate_ranges=None, threads=1):
        """
        Fits the parameters of the model to a data set. By default, learns a
        conditionally independent model. Additional unary dependencies can be
        set to be included in the constructor. Additional pairwise and
        higher-order dependencies can be included as an argument.

        Results are stored as a member named weights, instance of
        snorkel.learning.gen_learning.GenerativeModelWeights.

        :param L: M x N csr_AnnotationMatrix-type label matrix, where there are 
            M candidates labeled by N labeling functions (LFs)
        :param deps: collection of dependencies to include in the model, each 
                     element is a tuple of the form 
                     (LF 1 index, LF 2 index, dependency type),
                     see snorkel.learning.constants
        :param LF_acc_prior_weights: An N-element list of prior weights for the
            LF accuracies (log scale)
        :param LF_acc_prior_weight_default: Default prior for the weight of each 
            LF accuracy; if LF_acc_prior_weights is unset, each LF will have 
            this accuracy prior weight (log scale)
        :param labels: Optional ground truth labels
        :param label_prior_weight: The prior probability that the ground truth 
            labels (if provided) are correct (log scale)
        :param init_deps: initial weight for additional dependencies, except
                          class prior (log scale)
        :param init_class_prior: initial class prior (in log scale), note only
                                 used if class_prior=True in constructor
        :param epochs: number of training epochs
        :param step_size: gradient step size, default is 1 / L.shape[0]
        :param decay: multiplicative decay of step size,
                      step_size_(t+1) = step_size_(t) * decay
        :param reg_param: regularization strength
        :param reg_type: 1 = L1 regularization, 2 = L2 regularization
        :param verbose: whether to write debugging info to stdout
        :param truncation: number of iterations between truncation step for L1
                           regularization
        :param burn_in: number of burn-in samples to take before beginning
                        learning
        :param cardinality: number of possible classes; by default is inferred
            from the label matrix L
        :param timer: stopwatch for profiling, must implement start() and end()
        :param candidate_ranges: Optionally, a list of M sets of integer values,
            representing the possible categorical values that each of the M
            candidates can take. If a label is outside of this range throws an
            error. If None, then each candidate can take any value from 0 to
            cardinality.
        :param threads: the number of threads to use for sampling. Default is 1.
        """
        m, n = L.shape
        step_size = step_size or 0.0001

        # Check to make sure matrix is int-valued
        element_type = type(L[0,0])
        # HACK: Uncomment this!
        # if not element_type in [np.int64, np.int32, int]:
        #     raise ValueError("""Label matrix must have int-type elements, 
        #         but elements have type %s""" % element_type)

        # Automatically infer cardinality
        # Binary: Values in {-1, 0, 1} [Default]
        # Categorical: Values in {0, 1, ..., K}
        if cardinality is None:
            # If candidate_ranges is provided, use this to determine cardinality
            if candidate_ranges is not None:
                cardinality = max(map(max, candidate_ranges))
            else:
                # This is just an annoying hack for LIL sparse matrices...
                try:
                    lmax = L.max()
                except AttributeError:
                    lmax = L.tocoo().max()

                if lmax > 2:
                    cardinality = lmax
                elif lmax < 2:
                    cardinality = 2
                else:
                    raise ValueError(
                        "L.max() == %s, cannot infer cardinality." % lmax)
            print("Inferred cardinality: %s" % cardinality)
        self.cardinality = cardinality

        # Priors for LFs default to fixed prior value
        # NOTE: Setting default != 0.5 creates a (fixed) factor which increases
        # runtime (by ~0.5x that of a non-fixed factor)...
        if LF_acc_prior_weights is None:
            LF_acc_prior_weights = [LF_acc_prior_weight_default for _ in range(n)]
        else:
            LF_acc_prior_weights = list(copy(LF_acc_prior_weights))

        # LF weights are un-fixed
        is_fixed = [False for _ in range(n)]

        # If supervised labels are provided, add them as a fixed LF with prior
        # Note: For large L this column stack operation could be very
        # inefficient, can consider refactoring...
        if labels is not None:
            labels = labels.reshape(m, 1)
            L = sparse.hstack([L, labels])
            is_fixed.append(True)
            LF_acc_prior_weights.append(label_prior_weight)
            n += 1

        # Reduce overhead of tracking indices by converting L to a CSR sparse matrix.
        L = sparse.csr_matrix(L).copy()

        # If candidate_ranges is provided, remap the values of L using
        # candidate_ranges. This "scoped categorical" approach allows learning
        # and inference to be efficient even with very large cardinality, as
        # we only sample relevant values for each candidate. Also set
        # per-candidate cardinalities according to candidate_ranges if not None,
        # else as constant value.
        self.cardinalities = self.cardinality * np.ones(m, dtype=np.int64)
        self.candidate_ranges = candidate_ranges
        if self.candidate_ranges is not None:
            L, self.cardinalities, _ = self._remap_scoped_categoricals(L, 
                self.candidate_ranges)

        # Shuffle the data points, cardinalities, and candidate_ranges
        idxs = range(m)
        self.rng.shuffle(idxs)
        L = L[idxs, :]
        if candidate_ranges is not None:
            self.cardinalities = self.cardinalities[idxs]
            c_ranges_reshuffled = []
            for i in idxs:
                c_ranges_reshuffled.append(self.candidate_ranges[i])
            self.candidate_ranges = c_ranges_reshuffled

        # Compile factor graph
        self._process_dependency_graph(L, deps)
        weight, variable, factor, ftv, domain_mask, n_edges = self._compile(
            L, init_deps, init_class_prior, LF_acc_prior_weights, is_fixed, self.cardinalities)
        fg = NumbSkull(
            n_inference_epoch=0,
            n_learning_epoch=epochs, 
            stepsize=step_size,
            decay=decay,
            reg_param=reg_param,
            regularization=reg_type,
            truncation=truncation,
            quiet=(not verbose),
            verbose=verbose, 
            learn_non_evidence=True,
            burn_in=burn_in,
            nthreads=threads
        )
        fg.loadFactorGraph(weight, variable, factor, ftv, domain_mask, n_edges)

        if timer is not None:
            timer.start()
        fg.learning(out=False)
        if timer is not None:
            timer.end()
        self._process_learned_weights(L, fg, LF_acc_prior_weights, is_fixed)

        # Store info from factor graph
        if self.candidate_ranges is not None:
            self.cardinality_for_stats = int(max(self.cardinalities))
        else:
            self.cardinality_for_stats = self.cardinality
        self.learned_weights = fg.factorGraphs[0].weight_value
        weight, variable, factor, ftv, domain_mask, n_edges =\
            self._compile(sparse.coo_matrix((1, n), L.dtype), init_deps,
                init_class_prior, LF_acc_prior_weights, is_fixed,
                [self.cardinality_for_stats])

        variable["isEvidence"] = False
        weight["isFixed"] = True
        weight["initialValue"] = fg.factorGraphs[0].weight_value

        fg.factorGraphs = []
        fg.loadFactorGraph(weight, variable, factor, ftv, domain_mask, n_edges)

        self.fg = fg
        self.nlf = n
        self.cardinality = cardinality

    def _remap_scoped_categoricals(self, L_in, candidate_ranges):
        """
        Remap the values of each individual candidate so that they have dense
        support, returning the remapped label matrix, cardinalities, and
        inverse mapping.
        """
        L = L_in.copy()
        m, n = L.shape
        cardinalities = np.ones(m)
        mappings = []
        for i in range(m):
            c_range = candidate_ranges[i]

            # Confirm that the candidate range has only unique values
            assert len(c_range) == len(set(c_range))
            cardinalities[i] = len(c_range)

            # Create the inverse mapping
            mappings.append(dict([(a + 1, b) for a, b in enumerate(c_range)]))

            # Re-map the values of L[i, :]
            # Assumes L is csr_sparse format at this point
            for j in range(L[i].data.shape[0]):
                val = L[i].data[j]
                if val not in c_range:
                    raise ValueError("""Value {0} is not in supplied range 
                        for candidate at index {1}""".format(val, i))
                L[i, L[i].indices[j]] = c_range.index(val) + 1
        return L, cardinalities, mappings

    def learned_lf_stats(self):
        """
        Provides a summary of what the model has learned about the labeling
        functions. For each labeling function, estimates of the following
        are provided:

            Abstain
            Accuracy
            Coverage

            [Following are only available for binary tasks]
            True  Positive (TP)
            False Positive (FP)
            True  Negative (TN)
            False Negative (FN)

        For scoped categoricals, the information provided is for the maximum
        observed cardinality of any single data point.

        WARNING: This uses Gibbs sampling to estimate these values. This will
                 tend to mix poorly when there are many very accurate labeling
                 functions. In this case, this function will assume that the
                 classes are approximately balanced.
        """
        if self.fg is None:
            raise ValueError(
                "Must fit model with train() before computing diagnostics.")

        burnin = 500
        trials = 5000
        cardinality = self.cardinality_for_stats
        count = np.zeros((self.nlf, cardinality, cardinality + 1))

        for true_label in range(cardinality):
            for i in range(self.nlf + 1):
                self.fg.factorGraphs[0].var_value[0, i] = true_label
            self.fg.factorGraphs[0].inference(burnin, 0, True)
            for i in range(trials):
                self.fg.factorGraphs[0].inference(0, 1, True)
                y = self.fg.factorGraphs[0].var_value[0, 0]
                for j in range(self.nlf):
                    lf = self.fg.factorGraphs[0].var_value[0, j + 1]
                    count[j, y, lf] += 1

        count /= cardinality * trials

        # Compute summary stats to return to user
        stats = []
        for i in range(self.nlf):
            if cardinality == 2:
                tp = count[i, 1, 1]
                fp = count[i, 0, 1]
                tn = count[i, 0, 0]
                fn = count[i, 1, 0]
                coverage = 1 - (count[i, 0, 2] + count[i, 1, 2])
                stats.append({
                    "Precision": tp / (tp + fp),
                    "Recall": tp / count[i, 1, :].sum(),
                    "Accuracy": (tp + tn) / coverage,
                    "Coverage": coverage
                    })
            else:
                correct = sum([count[i, j, j] for j in range(cardinality)])
                coverage = 1 - sum([count[i, j, cardinality]
                    for j in range(cardinality)])
                stats.append({
                    "Accuracy": correct / coverage,
                    "Coverage": coverage
                })

        return DataFrame(stats)

    def marginals(self, L, candidate_ranges=None, batch_size=None):
        """
        Given an M x N label matrix, returns marginal probabilities for each
        candidate, depending on classification setting:

            - Binary: Returns M-dim array representing the marginal probability
                of each candidate being True

            - Categorical (cardinality = K): Returns M x K dense matrix
                representing the marginal probabilities of each candidate being
                each class.

            - Scoped Categorical (cardinality = K, cardinality_ranges not None):
                Returns an M x K *sparse* matrix of marginals.

        In the categorical setting, the K values (columns in the marginals
        matrix) correspond to indices of the Candidate values defined.
        """
        m, n = L.shape
        if self.weights is None:
            raise ValueError("""Must fit model with train() before computing 
                marginal probabilities.""")

        # Binary classification setting
        if self.cardinality == 2:
            marginals = np.ndarray(L.shape[0], dtype=np.float64)

            for i in range(L.shape[0]):
                logp_true = self.weights.class_prior
                logp_false = -1 * self.weights.class_prior

                l_i = L[i].tocoo()

                for l_index1 in range(l_i.nnz):
                    data_j, j = l_i.data[l_index1], l_i.col[l_index1]
                    if data_j == 1:
                        logp_true  += self.weights.lf_accuracy[j]
                        logp_false -= self.weights.lf_accuracy[j]
                        logp_true  += self.weights.lf_class_propensity[j]
                        logp_false -= self.weights.lf_class_propensity[j]
                    elif data_j == -1:
                        logp_true  -= self.weights.lf_accuracy[j]
                        logp_false += self.weights.lf_accuracy[j]
                        logp_true  += self.weights.lf_class_propensity[j]
                        logp_false -= self.weights.lf_class_propensity[j]
                    else:
                        ValueError("""Illegal value at %d, %d: %d.
                            Must be in {-1, 0, 1}.""" % (i, j, data_j))

                    for l_index2 in range(l_i.nnz):
                        data_k, k = l_i.data[l_index2], l_i.col[l_index2]
                        if j != k:
                            if data_j == -1 and data_k == 1:
                                logp_true += self.weights.dep_fixing[j, k]
                            elif data_j == 1 and data_k == -1:
                                logp_false += self.weights.dep_fixing[j, k]

                            if data_j == 1 and data_k == 1:
                                logp_true += self.weights.dep_reinforcing[j, k]
                            elif data_j == -1 and data_k == -1:
                                logp_false += self.weights.dep_reinforcing[j, k]

                marginals[i] = 1 / (1 + np.exp(logp_false - logp_true))
            return marginals

        # Categorical setting
        else:
            all_marginals = []

            # Handle the scoped categorical case, otherwise get cardinalities
            # from self.cardinality
            if candidate_ranges is not None:
                L, cardinalities, mappings = self._remap_scoped_categoricals(L, 
                    candidate_ranges)
            else:
                cardinalities = self.cardinality * np.ones(m)

            # Get the marginal (posterior) probability for each candidate
            for i in range(m):
                cardinality = int(cardinalities[i])
                marginals = np.zeros(cardinality, dtype=np.float64)
                # NB: class priors not currently available for categoricals
                l_i = L[i].tocoo()
                for l_index1 in range(l_i.nnz):
                    data_j, j = l_i.data[l_index1], l_i.col[l_index1]
                    if (data_j != 0):
                        if not 1 <= data_j <= cardinality:
                            raise ValueError(
                                """Illegal value at %d, %d: %d. Must be in 0 to 
                                %d.""" % (i, j, data_j, cardinality))
                        # NB: LF class propensity not currently available
                        # for categoricals
                        marginals[int(data_j - 1)] += \
                            2 * self.weights.lf_accuracy[j]
                            
                # NB: fixing and reinforcing not available for categoricals
                # Get softmax
                exps = np.exp(marginals)
                marginals = exps / exps.sum()
                all_marginals.append(marginals)

            # If candidate_ranges not None, remap back to original values and
            # return as sparse matrix
            if candidate_ranges is not None:
                M = sparse.coo_matrix((m, self.cardinality), dtype=np.float64)
                for i, marginals in enumerate(all_marginals):
                    for j, p in enumerate(marginals):
                        M[i, mappings[i][j]] = p
            else:
                M = np.vstack(all_marginals)
            return M

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

    def _compile(self, L, init_deps, init_class_prior, LF_acc_prior_weights, is_fixed, cardinalities):
        """Compiles a generative model based on L and the current labeling function
        dependencies.
        """
        m, n = L.shape

        n_weights = 1 if self.class_prior else 0

        self.hasPrior = [i != 0 for i in LF_acc_prior_weights]
        nPrior = sum(self.hasPrior)
        nUnFixed = sum([not i for i in is_fixed])

        n_weights += nPrior
        n_weights += nUnFixed
        for optional_name in GenerativeModel.optional_names:
            if getattr(self, optional_name):
                n_weights += n
        for dep_name in GenerativeModel.dep_names:
            n_weights += getattr(self, dep_name).getnnz()

        n_vars = m * (n + 1)
        n_factors = m * n_weights

        n_edges = 1 if self.class_prior else 0
        n_edges += 2 * (nPrior + nUnFixed)
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
            weight[0]['initialValue'] = np.float64(init_class_prior)
            w_off = 1
        else:
            w_off = 0

        for i in range(n):
            # Prior on LF acc
            if self.hasPrior[i]:
                weight[w_off]['isFixed'] = True
                weight[w_off]['initialValue'] = LF_acc_prior_weights[i]
                w_off += 1
            # Learnable acc for LF
            if (not is_fixed[i]):
                weight[w_off]['isFixed'] = False

                # Note: Because we're not doing exact gradient descent, don't
                # need to add any random noise to initial values here
                # Setting to 0 = setting to prior value
                weight[w_off]['initialValue'] = np.float64(0)
                w_off += 1

        for i in range(w_off, weight.shape[0]):
            weight[i]['isFixed'] = False
            weight[i]['initialValue'] = np.float64(init_deps)

        #
        # Compiles variable matrix
        #
        # Internal representation:
        #   True Class:         0 to (cardinality - 1) are the classes
        #   Labeling functions: 0 to (cardinality - 1) are the classes
        #                       cardinality is abstain
        # Candidates (variables)
        for i in range(m):
            variable[i]['isEvidence'] = False
            variable[i]['initialValue'] = self.rng.randint(cardinalities[i])
            variable[i]["dataType"] = 0
            variable[i]["cardinality"] = cardinalities[i]

        # LF label variables -- initial loop to set all variables
        for i in range(m):
            for j in range(n):
                index = m + n * i + j
                variable[index]["isEvidence"] = 1
                variable[index]["dataType"] = 0
                variable[index]["cardinality"] = cardinalities[i] + 1
                
                # Default to abstain
                variable[index]["initialValue"] = cardinalities[i]

        # LF labels -- now set the non-zero labels
        L_coo = L.tocoo()
        for L_index in range(L_coo.nnz):
            data, i, j = L_coo.data[L_index], L_coo.row[L_index], L_coo.col[L_index]
            index = m + n * i + j

            # Note: Here we need to use the overall cardinality to handle, since
            # with candidate_ranges not None and self.cardinality > 2, some
            # candidates could have cardinality == 2...
            if (self.cardinality == 2):
                if data == 1:
                    variable[index]["initialValue"] = 1
                elif data == 0:
                    variable[index]["initialValue"] = 2
                elif data == -1:
                    variable[index]["initialValue"] = 0
                else:
                    raise ValueError("Invalid labeling function output in cell (%d, %d): %d. "
                                     "Valid values are 1, 0, and -1. " % (i, j, data))
            else:
                if data == 0:
                    variable[index]["initialValue"] = cardinalities[i]
                elif 1 <= data <= cardinalities[i]:
                    variable[index]["initialValue"] = data - 1
                else:
                    raise ValueError("Invalid labeling function output in cell (%d, %d): %d. "
                                     "Valid values are 0 to %d. " % (i, j, data, self.cardinalities[i]))

        #
        # Compiles factor and ftv matrices
        #
        # Class prior
        if self.class_prior:
            if self.cardinality != 2:
                raise NotImplementedError("Class Prior not implemented for categorical classes.")
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
        nfactors_for_lf = [(int(self.hasPrior[i]) + int(not is_fixed[i])) for i in range(n)]
        f_off, ftv_off, w_off = self._compile_output_factors(L, factor, f_off, ftv, ftv_off, w_off, "DP_GEN_LF_ACCURACY",
                                                             (lambda m, n, i, j: i, lambda m, n, i, j: m + n * i + j), nfactors_for_lf)

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
                if optional_name != 'lf_propensity' and self.cardinality != 2:
                    raise NotImplementedError(optional_name + " not implemented for categorical classes.")
                f_off, ftv_off, w_off = self._compile_output_factors(L, factor, f_off, ftv, ftv_off, w_off,
                                                                     optional_name_map[optional_name][0],
                                                                     optional_name_map[optional_name][1])

        # Factors for labeling function dependencies
        dep_name_map = {
            'dep_similar':
                ('DP_GEN_DEP_SIMILAR', (
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

        CATEGORICAL_DEPS = ['dep_similar', 'dep_exclusive']
        for dep_name in GenerativeModel.dep_names:
            mat = getattr(self, dep_name)
            if mat.nnz > 0:
                if dep_name not in CATEGORICAL_DEPS and self.cardinality != 2:
                    raise NotImplementedError(
                        dep_name + " not implemented for categorical classes.")
                for i in range(len(mat.data)):
                    f_off, ftv_off, w_off = self._compile_dep_factors(L, factor, 
                        f_off, ftv, ftv_off, w_off, mat.row[i], mat.col[i],
                        dep_name_map[dep_name][0], dep_name_map[dep_name][1])

        return weight, variable, factor, ftv, domain_mask, n_edges

    def _compile_output_factors(self, L, factors, factors_offset, ftv, 
        ftv_offset, weight_offset, factor_name, vid_funcs,
        nfactors_for_lf=None):
        """
        Compiles factors over the outputs of labeling functions, i.e., for which
        there is one weight per labeling function and one factor per labeling 
        function-candidate pair.
        """
        m, n = L.shape

        if nfactors_for_lf == None:
            nfactors_for_lf = [1 for i in range(n)]

        factors_index = factors_offset
        ftv_index = ftv_offset
        for i in range(m):
            w_off = weight_offset
            for j in range(n):
                for k in range(nfactors_for_lf[j]):
                    factors[factors_index]["factorFunction"] = FACTORS[factor_name]
                    factors[factors_index]["weightId"] = w_off
                    factors[factors_index]["featureValue"] = 1
                    factors[factors_index]["arity"] = len(vid_funcs)
                    factors[factors_index]["ftv_offset"] = ftv_index

                    factors_index += 1
                    w_off += 1

                    for vid_func in vid_funcs:
                        ftv[ftv_index]["vid"] = vid_func(m, n, i, j)
                        ftv_index += 1

        return factors_index, ftv_index, w_off

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

    def _process_learned_weights(self, L, fg, LF_acc_prior_weights, is_fixed):
        _, n = L.shape

        w = fg.getFactorGraph().getWeights()
        weights = GenerativeModelWeights(n)

        if self.class_prior:
            weights.class_prior = w[0]
            w_off = 1
        else:
            w_off = 0

        weights.lf_accuracy = np.zeros((n,))
        for i in range(n):
            # Prior on LF acc
            if self.hasPrior[i]:
                weights.lf_accuracy[i] += w[w_off]
                w_off += 1
            # Learnable acc for LF
            if (not is_fixed[i]):
                weights.lf_accuracy[i] += w[w_off]
                w_off += 1

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

    def save(self, model_name=None, save_dir='checkpoints', verbose=True):
        """Save current model."""
        model_name = model_name or self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save generative model weights
        save_path = os.path.join(save_dir, "{0}.weights.pkl".format(model_name))
        with open(save_path, 'wb') as f:
            dump(self.weights, f)

        # Save other model hyperparameters needed to rebuild model
        save_path2 = os.path.join(save_dir, "{0}.hps.pkl".format(model_name))
        with open(save_path2, 'wb') as f:
            dump({
                'cardinality': self.cardinality,
                'cardinality_for_stats': self.cardinality_for_stats
            }, f)

        if verbose:
            print("[{0}] Model saved as <{1}>.".format(self.name, model_name))

    def load(self, model_name=None, save_dir='checkpoints', verbose=True):
        """Load model."""
        model_name = model_name or self.name
        save_path = os.path.join(save_dir, "{0}.weights.pkl".format(model_name))
        with open(save_path, 'rb') as f:
            self.weights = load(f)
        save_path2 = os.path.join(save_dir, "{0}.hps.pkl".format(model_name))
        with open(save_path2, 'rb') as f:
            hps = load(f)
            for k, v in hps.iteritems():
                setattr(self, k, v)
        if verbose:
            print("[{0}] Model <{1}> loaded.".format(self.name, model_name))

    def _preprocess_data(self, X):
        """Generic preprocessing subclass; may be called by external methods."""
        return X


class GenerativeModelWeights(object):

    def __init__(self, n):
        self.n = n
        self.class_prior = 0.0
        self.lf_accuracy = np.zeros(n, dtype=np.float64)
        for optional_name in GenerativeModel.optional_names:
            setattr(self, optional_name, np.zeros(n, dtype=np.float64))

        for dep_name in GenerativeModel.dep_names:
            setattr(self, dep_name, sparse.lil_matrix((n, n), dtype=np.float64))

    def is_sign_sparsistent(self, other, threshold=0.1):
        if self.n != other.n:
            raise ValueError("Dimension mismatch. %d versus %d" % (self.n, other.n))

        if not self._weight_is_sign_sparsitent(self.class_prior, other.class_prior, threshold):
            return False

        for i in range(self.n):
            if not self._weight_is_sign_sparsitent(
                    self.lf_accuracy[i], other.lf_accuracy[i], threshold):
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


@jit
def set_numba_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
