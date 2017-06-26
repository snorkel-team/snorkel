import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import warnings
from itertools import product

from pandas import DataFrame

matplotlib.use('Agg')
warnings.filterwarnings("ignore", module="matplotlib")


def reshape_marginals(marginals):
    """Returns correctly shaped marginals as np array"""
    # Make sure training marginals are a numpy array first
    try:
        shape = marginals.shape
    except:
        marginals = np.array(marginals)
        shape = marginals.shape
    
    # Set cardinality + marginals in proper format for binary v. categorical
    if len(shape) != 1:
        # If k = 2, make sure is M-dim array
        if shape[1] == 2:
            marginals = marginals[:,1].reshape(-1)
    return marginals


class LabelBalancer(object):
    def __init__(self, y):
        """Utility class to rebalance training labels
        For example, to get the indices of a training set
        with labels y and around 90 percent negative examples,
            LabelBalancer(y).get_train_idxs(rebalance=0.1)
        """
        self.y = np.ravel(y)
    
    def _get_pos(self, split):
        return np.where(self.y > (split + 1e-6))[0]

    def _get_neg(self, split):
        return np.where(self.y < (split - 1e-6))[0]
    
    def _try_frac(self, m, n, pn):
        # Return (a, b) s.t. a <= m, b <= n
        # and b / a is as close to pn as possible
        r = int(round(float(pn * m) / (1.0-pn)))
        s = int(round(float((1.0-pn) * n) / pn))
        return (m,r) if r <= n else ((s,n) if s <= m else (m,n))

    def _get_counts(self, nneg, npos, frac_pos):
        if frac_pos > 0.5:
            return self._try_frac(nneg, npos, frac_pos)
        else:
            return self._try_frac(npos, nneg, 1.0-frac_pos)[::-1]

    def get_train_idxs(self, rebalance=False, split=0.5):
        """Get training indices based on @y
            @rebalance: bool or fraction of positive examples desired
                        If True, default fraction is 0.5. If False no balancing.
            @split: Split point for positive and negative classes
        """
        pos, neg = self._get_pos(split), self._get_neg(split)
        if rebalance:
            if len(pos) == 0:
                raise ValueError("No positive labels.")
            if len(neg) == 0:
                raise ValueError("No negative labels.")
            p = 0.5 if rebalance == True else rebalance
            n_neg, n_pos = self._get_counts(len(neg), len(pos), p)
            pos = np.random.choice(pos, size=n_pos, replace=False)
            neg = np.random.choice(neg, size=n_neg, replace=False)
        idxs = np.concatenate([pos, neg])
        np.random.shuffle(idxs)
        return idxs


class Scorer(object):
    """Abstract type for scorers"""
    def __init__(self, test_candidates, test_labels, gold_candidate_set=None):
        """
        :param test_candidates: A *list of Candidates* corresponding to 
            test_labels
        :param test_labels: A *csrLabelMatrix* of ground truth labels for the 
            test candidates
        :param gold_candidate_set: (optional) A *CandidateSet* containing the 
            full set of gold labeled candidates
        """
        self.test_candidates    = test_candidates
        self.test_labels        = test_labels
        self.gold_candidate_set = gold_candidate_set

    def _get_cardinality(self, marginals):
        """Get the cardinality based on the marginals returned by the model."""
        if len(marginals.shape) == 1 or marginals.shape[1] < 3:
            cardinality = 2
        else:
            cardinality = marginals.shape[1]
        return cardinality

    def score(self, test_marginals, display=True, **kwargs):
        cardinality = self._get_cardinality(test_marginals)
        if cardinality == 2:
            return self._score_binary(test_marginals, **kwargs)
        else:
            return self._score_categorical(test_marginals, **kwargs)

    def _score_binary(self, test_marginals, train_marginals=None, b=0.5, 
        set_unlabeled_as_neg=True, display=True):
        raise NotImplementedError()

    def _score_categorical(self, test_marginals, train_marginals=None, 
        display=True):
        raise NotImplementedError()

    def summary_score(self, test_marginals, **kwargs):
        """Return the F1 score (for binary) or accuracy (for categorical)."""
        raise NotImplementedError()



class MentionScorer(Scorer):
    """Scorer for mention level assessment"""
    def _score_binary(self, test_marginals, train_marginals=None, b=0.5,
        set_unlabeled_as_neg=True, set_at_thresh_as_neg=True, display=True,
        **kwargs):
        """
        Return scoring metric for the provided marginals, as well as candidates
        in error buckets.

        :param test_marginals: array of marginals for test candidates
        :param train_marginals (optional): array of marginals for training 
            candidates
        :param b: threshold for labeling
        :param set_unlabeled_as_neg: set test labels at the decision threshold 
            of b as negative labels
        :param set_at_b_as_neg: set marginals at the decision threshold exactly
            as negative predictions
        :param display: show calibration plots?
        """
        test_label_array = []
        tp = set()
        fp = set()
        tn = set()
        fn = set()

        for i, candidate in enumerate(self.test_candidates):
            # Handle either a LabelMatrix or else assume test_labels array is in
            # correct order i.e. same order as test_candidates
            try:
                test_label_index = self.test_labels.get_row_index(candidate)
                test_label = self.test_labels[test_label_index, 0]
            except AttributeError:
                test_label = self.test_labels[i]

            # Set unlabeled examples to -1 by default
            if test_label == 0 and set_unlabeled_as_neg:
                test_label = -1
          
            # Bucket the candidates for error analysis
            test_label_array.append(test_label)
            if test_label != 0:
                if test_marginals[i] > b:
                    if test_label == 1:
                        tp.add(candidate)
                    else:
                        fp.add(candidate)
                elif test_marginals[i] < b or set_at_thresh_as_neg:
                    if test_label == -1:
                        tn.add(candidate)
                    else:
                        fn.add(candidate)
        if display:

            # Calculate scores unadjusted for TPs not in our candidate set
            print_scores(len(tp), len(fp), len(tn), len(fn), 
                title="Scores (Un-adjusted)")

            # If gold candidate set is provided calculate recall-adjusted scores
            if self.gold_candidate_set is not None:
                gold_fn = [c for c in self.gold_candidate_set
                    if c not in self.test_candidates]
                print "\n"
                print_scores(len(tp), len(fp), len(tn), len(fn)+len(gold_fn), 
                    title="Corpus Recall-adjusted Scores")

            # If training and test marginals provided print calibration plots
            if train_marginals is not None and test_marginals is not None:
                print "\nCalibration plot:"
                calibration_plots(train_marginals, test_marginals, 
                    np.asarray(test_label_array))
        return tp, fp, tn, fn

    def _score_categorical(self, test_marginals, train_marginals=None,
        display=True, **kwargs):
        """
        Return scoring metric for the provided marginals, as well as candidates
        in error buckets.

        :param test_marginals: array of marginals for test candidates
        :param train_marginals (optional): array of marginals for training 
            candidates
        :param display: show calibration plots?
        """
        test_label_array = []
        correct = set()
        incorrect = set()

        # Get predictions
        test_pred = test_marginals.argmax(axis=1) + 1

        # Bucket the candidates for error analysis
        for i, candidate in enumerate(self.test_candidates):
            # Handle either a LabelMatrix or else assume test_labels array is in
            # correct order i.e. same order as test_candidates
            try:
                test_label_index = self.test_labels.get_row_index(candidate)
                test_label = self.test_labels[test_label_index, 0]
            except AttributeError:
                test_label = self.test_labels[i]  
            test_label_array.append(test_label)
            if test_label != 0:
                if test_pred[i] == test_label:
                    correct.add(candidate)
                else:
                    incorrect.add(candidate)
        if display:
            nc, ni = len(correct), len(incorrect)
            print "Accuracy:", nc / float(nc + ni)

            # If gold candidate set is provided calculate recall-adjusted scores
            if self.gold_candidate_set is not None:
                gold_missed = [c for c in self.gold_candidate_set
                    if c not in self.test_candidates]
                print "Coverage:", (nc + ni) / (nc + ni + len(gold_missed))
        return correct, incorrect

    def summary_score(self, test_marginals, **kwargs):
        """
        Return the F1 score (for binary) or accuracy (for categorical).
        Also return the label as second argument.
        """
        error_sets = self.score(test_marginals, display=False, **kwargs)
        if len(error_sets) == 4:
            _, _, f1 = binary_scores_from_counts(*map(len, error_sets))
            return f1, "F1 Score"
        else:
            nc, ninc = map(len, error_sets)
            return nc / float(nc + ninc), "Accuracy"


def binary_scores_from_counts(ntp, nfp, ntn, nfn):
    """
    Precision, recall, and F1 scores from counts of TP, FP, TN, FN.
    Example usage:
        p, r, f1 = binary_scores_from_counts(*map(len, error_sets))
    """
    prec = ntp / float(ntp + nfp) if ntp + nfp > 0 else 0.0
    rec  = ntp / float(ntp + nfn) if ntp + nfn > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def print_scores(ntp, nfp, ntn, nfn, title='Scores'):
    prec, rec, f1 = binary_scores_from_counts(ntp, nfp, ntn, nfn)
    pos_acc = ntp / float(ntp + nfn) if ntp + nfn > 0 else 0.0
    neg_acc = ntn / float(ntn + nfp) if ntn + nfp > 0 else 0.0
    print("========================================")
    print(title)
    print("========================================")
    print("Pos. class accuracy: {:.3}".format(pos_acc))
    print("Neg. class accuracy: {:.3}".format(neg_acc))
    print("Precision            {:.3}".format(prec))
    print("Recall               {:.3}".format(rec))
    print("F1                   {:.3}".format(f1))
    print("----------------------------------------")
    print("TP: {} | FP: {} | TN: {} | FN: {}".format(ntp, nfp, ntn, nfn))
    print("========================================\n")


def plot_prediction_probability(probs):
    plt.hist(probs, bins=20, normed=False, facecolor='blue')
    plt.xlim((0,1.025))
    plt.xlabel("Probability")
    plt.ylabel("# Predictions")


def plot_accuracy(probs, ground_truth):
    x = 0.1 * np.array(range(11))
    bin_assign = [x[i] for i in np.digitize(probs, x)-1]
    correct = ((2*(probs >= 0.5) - 1) == ground_truth)
    correct_prob = np.array([np.mean(correct[bin_assign == p]) for p in x])
    xc = x[np.isfinite(correct_prob)]
    correct_prob = correct_prob[np.isfinite(correct_prob)]
    plt.plot(x, np.abs(x-0.5) + 0.5, 'b--', xc, correct_prob, 'ro-')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel("Probability")
    plt.ylabel("Accuracy")


def calibration_plots(train_marginals, test_marginals, gold_labels=None):
    """Show classification accuracy and probability histogram plots"""
    n_plots = 3 if gold_labels is not None else 1
    
    # Whole set histogram
    plt.subplot(1,n_plots,1)
    plot_prediction_probability(train_marginals)
    plt.title("(a) # Predictions (training set)")

    if gold_labels is not None:

        # Hold-out histogram
        plt.subplot(1,n_plots,2)
        plot_prediction_probability(test_marginals)
        plt.title("(b) # Predictions (test set)")

        # Classification bucket accuracy
        plt.subplot(1,n_plots,3)
        plot_accuracy(test_marginals, gold_labels)
        plt.title("(c) Accuracy (test set)")
    plt.show()


def grid_search_plot(w_fit, mu_opt, f1_opt):
    """Plot validation set performance for logistic regression regularization"""
    mu_seq = sorted(w_fit.keys())
    p = np.ravel([w_fit[mu].P for mu in mu_seq])
    r = np.ravel([w_fit[mu].R for mu in mu_seq])
    f1 = np.ravel([w_fit[mu].F1 for mu in mu_seq])
    nnz = np.ravel([np.sum(w_fit[mu].w != 0) for mu in mu_seq])    

    fig, ax1 = plt.subplots()
    
    # Plot spread
    ax1.set_xscale('log', nonposx='clip')    
    ax1.scatter(mu_opt, f1_opt, marker='*', color='purple', s=500,
                zorder=10, label="Maximum F1: mu={}".format(mu_opt))
    ax1.plot(mu_seq, f1, 'o-', color='red', label='F1 score')
    ax1.plot(mu_seq, p, 'o--', color='blue', label='Precision')
    ax1.plot(mu_seq, r, 'o--', color='green', label='Recall')
    ax1.set_xlabel('log(penalty)')
    ax1.set_ylabel('F1 score/Precision/Recall')
    ax1.set_ylim(-0.04, 1.04)
    for t1 in ax1.get_yticklabels():
      t1.set_color('r')
    
    # Plot nnz
    ax2 = ax1.twinx()
    ax2.plot(mu_seq, nnz, '.:', color='gray', label='Sparsity')
    ax2.set_ylabel('Number of non-zero coefficients')
    ax2.set_ylim(-0.01*np.max(nnz), np.max(nnz)*1.01)
    for t2 in ax2.get_yticklabels():
      t2.set_color('gray')
    
    # Shrink plot for legend
    box1 = ax1.get_position()
    ax1.set_position(
        [box1.x0, box1.y0+box1.height*0.1, box1.width, box1.height*0.9]
    )
    box2 = ax2.get_position()
    ax2.set_position(
        [box2.x0, box2.y0+box2.height*0.1, box2.width, box2.height*0.9]
    )
    plt.title("Validation for logistic regression learning")
    lns1, lbs1 = ax1.get_legend_handles_labels()
    lns2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1+lns2, lbs1+lbs2, loc='upper center', scatterpoints=1,
        bbox_to_anchor=(0.5,-0.05),  fontsize=10, markerscale=0.5)
    plt.show()

    
class Hyperparameter(object):
    """Base class for a grid search parameter"""
    def __init__(self, name):
        self.name = name
    
    def get_all_values(self):
        raise NotImplementedError()
    
    def draw_values(self, n):
        # Multidim parameters can't use choice directly
        v = self.get_all_values()
        return [v[int(i)] for i in np.random.choice(len(v), n)]

    
class ListParameter(Hyperparameter):
    """List of parameter values for searching"""
    def __init__(self, name, parameter_list):
        self.parameter_list = np.array(parameter_list)
        super(ListParameter, self).__init__(name)
    
    def get_all_values(self):
        return self.parameter_list

    
class RangeParameter(Hyperparameter):
    """
    Range of parameter values for searching.
    min_value and max_value are the ends of the search range
    If log_base is specified, scale the search range in the log base
    step is range step size or exponent step size
    """
    def __init__(self, name, v1, v2, step=1, log_base=None):
        self.min_value = min(v1, v2)
        self.max_value = max(v1, v2)
        self.step = step
        self.log_base = log_base
        super(RangeParameter, self).__init__(name)
        
    def get_all_values(self):
        if self.log_base:
            min_exp = math.log(self.min_value, self.log_base)
            max_exp = math.log(self.max_value, self.log_base)
            exps = np.arange(min_exp, max_exp + self.step, step=self.step)
            return np.power(self.log_base, exps)
        return np.arange(
            self.min_value, self.max_value + self.step, step=self.step
        )
        

class GridSearch(object):
    """
    Runs hyperparameter grid search over a model object with train and score methods,
    training data (X), and training_marginals
    Selects based on maximizing F1 score on a supplied validation set
    Specify search space with Hyperparameter arguments
    """
    def __init__(self, session, model, X, training_marginals,
        parameters, scorer=MentionScorer):
        self.session            = session
        self.model              = model
        self.X                  = X
        self.training_marginals = training_marginals
        self.params             = parameters
        self.param_names        = [param.name for param in parameters]
        self.scorer             = scorer
        
    def search_space(self):
        return product(*[param.get_all_values() for param in self.params])

    def fit(self, X_validation, validation_labels, b=0.5,
        set_unlabeled_as_neg=True, validation_kwargs={}, **model_hyperparams):
        """
        Basic method to start grid search, returns DataFrame table of results
          b specifies the positive class threshold for calculating f1
          set_unlabeled_as_neg is used to decide class of unlabeled cases for f1
          Non-search parameters are set using model_hyperparamters
        """
        # Iterate over the param values
        run_stats       = []
        run_score_opt   = -1.0
        base_model_name = self.model.name
        model_k         = 0
        for k, param_vals in enumerate(self.search_space()):
            model_name = '{0}_{1}'.format(base_model_name, model_k)
            model_k += 1
            # Set the new hyperparam configuration to test
            for pn, pv in zip(self.param_names, param_vals):
                model_hyperparams[pn] = pv
            print("=" * 60)
            print("[%d] Testing %s" % (k+1, ', '.join([
                "%s = %0.2e" % (pn,pv)
                for pn,pv in zip(self.param_names, param_vals)
            ])))
            print("=" * 60)
            # Train the model
            self.model.train(
                self.X, self.training_marginals, **model_hyperparams)
            # Test the model
            run_scores = self.model.score(X_validation, validation_labels, b=b,
                set_unlabeled_as_neg=set_unlabeled_as_neg)
            if self.model.cardinality > 2:
                run_score, run_score_label = run_scores, "Accuracy"
                run_scores = [run_score]
            else:
                run_score, run_score_label = run_scores[-1], "F1 Score"
            # Add scores to running stats, print, and set as optimal if best
            print("[{0}] {1}: {2}".format(self.model.name, run_score_label,
                run_score))
            run_stats.append(list(param_vals) + list(run_scores))
            if run_score > run_score_opt or k == 0:
                self.model.save(model_name=model_name)
                opt_model = model_name
                run_score_opt = run_score
        # Set optimal parameter in the learner model
        self.model.load(opt_model)
        # Return DataFrame of scores
        run_score_labels = ['Acc.'] if self.model.cardinality > 2 else \
            ['Prec.', 'Rec.', 'F1']
        sort_by = 'Acc.' if self.model.cardinality > 2 else 'F1'
        self.results = DataFrame.from_records(
            run_stats, columns=self.param_names + run_score_labels
        ).sort_values(by=sort_by, ascending=False)
        return self.results
    
    
class RandomSearch(GridSearch):
    def __init__(self, session, model, X, training_marginals, parameters, n=10, **kwargs):
        """Search a random sample of size n from a parameter grid"""
        self.n = n
        super(RandomSearch, self).__init__(
            session, model, X, training_marginals, parameters, **kwargs
        )

        print("Initialized RandomSearch search of size {0}. Search space size = {1}.".format(
            self.n, np.product([len(param.get_all_values()) for param in self.params]))
        )
        
    def search_space(self):
        return zip(*[param.draw_values(self.n) for param in self.params])


def sparse_abs(X):
    """Element-wise absolute value of sparse matrix- avoids casting to dense matrix!"""
    X_abs = X.copy()
    if not sparse.issparse(X):
        return abs(X_abs)
    if sparse.isspmatrix_csr(X) or sparse.isspmatrix_csc(X):
        X_abs.data = np.abs(X_abs.data)
    elif sparse.isspmatrix_lil(X):
        X_abs.data = np.array([np.abs(L) for L in X_abs.data])
    else:
        raise ValueError("Only supports CSR/CSC and LIL matrices")
    return X_abs


def candidate_coverage(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates which have > 0 (non-zero) labels.**
    """
    return np.where(sparse_abs(L).sum(axis=1) != 0, 1, 0).sum() / float(L.shape[0])


def LF_coverage(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF labels.**
    """
    return np.ravel(sparse_abs(L).sum(axis=0) / float(L.shape[0]))


def candidate_overlap(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates which have > 1 (non-zero) labels.**
    """
    return np.where(sparse_abs(L).sum(axis=1) > 1, 1, 0).sum() / float(L.shape[0])


def LF_overlaps(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _overlaps with other LFs on_.**
    """
    L_abs = sparse_abs(L)
    return np.ravel(np.where(L_abs.sum(axis=1) > 1, 1, 0).T * L_abs / float(L.shape[0]))


def candidate_conflict(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates which have > 1 (non-zero) labels _which are not equal_.**
    """
    return np.where(sparse_abs(L).sum(axis=1) != sparse_abs(L.sum(axis=1)), 1, 0).sum() / float(L.shape[0])


def LF_conflicts(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _conflicts with other LFs on_.**
    """
    L_abs = sparse_abs(L)
    return np.ravel(np.where(L_abs.sum(axis=1) != sparse_abs(L.sum(axis=1)), 1, 0).T * L_abs / float(L.shape[0]))


def LF_accuracies(L, labels):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate, and labels {-1,1}
    Return the accuracy of each LF w.r.t. these labels
    """
    return np.ravel(0.5*(L.T.dot(labels) / sparse_abs(L).sum(axis=0) + 1))


def training_set_summary_stats(L, return_vals=True, verbose=False):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return simple summary statistics
    """
    N, M = L.shape
    coverage, overlap, conflict = candidate_coverage(L), candidate_overlap(L), candidate_conflict(L)
    if verbose:
        print("=" * 60)
        print("LF Summary Statistics: %s LFs applied to %s candidates" % (M, N))
        print("-" * 60)
        print("Coverage (candidates w/ > 0 labels):\t\t%0.2f%%" % (coverage*100,))
        print("Overlap (candidates w/ > 1 labels):\t\t%0.2f%%" % (overlap*100,))
        print("Conflict (candidates w/ conflicting labels):\t%0.2f%%" % (conflict*100,))
        print("=" * 60)
    if return_vals:
        return coverage, overlap, conflict


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
  # Threshold to prevent float rollover into infinity/zero
  l[l > 25] = 25
  l[l < -25] = -25
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


def transform_sample_stats(Xt, t, f, Xt_abs=None):
  """
  Here we calculate the expected accuracy of each LF/feature
  (corresponding to the rows of X) wrt to the distribution of samples S:

    E_S[ accuracy_i ] = E_(t,f)[ \frac{TP + TN}{TP + FP + TN + FN} ]
                      = \frac{X_{i|x_{ij}>0}*t - X_{i|x_{ij}<0}*f}{t+f}
                      = \frac12\left(\frac{X*(t-f)}{t+f} + 1\right)
  """
  if Xt_abs is None:
    Xt_abs = sparse_abs(Xt) if sparse.issparse(Xt) else abs(Xt)
  n_pred = Xt_abs.dot(t+f)
  m = (1. / (n_pred + 1e-8)) * (Xt.dot(t) - Xt.dot(f))
  p_correct = (m + 1) / 2
  return p_correct, n_pred
