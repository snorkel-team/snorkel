# Base Python
import cPickle, json, os, sys, warnings
from collections import defaultdict, OrderedDict, namedtuple
import lxml.etree as et

# Scientific modules
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", module="matplotlib")
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from itertools import product
from pandas import DataFrame

def score(test_candidates, test_labels, test_pred, gold_candidate_set, train_marginals=None, test_marginals=None):
    '''
    Compute score with true recall
    
    :param candidates       candidate set
    :param candidates_gold  candidate set gold (true candidate)
    :param gold             true set gold
    :param pred             model predictions 
    
    '''
    # false negatives from complete gold set (missing from candidate set)
    gold_fn = [c for c in gold_candidate_set if c not in test_candidates]

    # Print calibration plots
    if train_marginals is not None and test_marginals is not None:
        print "Calibration plot:"
        calibration_plots(train_marginals, test_marginals, test_labels)
    
    # candidate match sets
    _, _, _, m_tp, m_fp, m_tn, m_fn, m_n_t = test_scores(test_pred, test_labels, return_vals=True, verbose=True)

    # model scores, augmented by missing FN set
    prec = m_tp / float(m_tp + m_fp)
    rec  = m_tp / float(m_tp + m_fn + len(gold_fn))
    f1 = 2.0 * (prec * rec) / (prec + rec)
    
    # Corrected FN
    fn = m_fn + len(gold_fn)

    print "========================================"
    print "Recall-corrected Noise-aware Model"
    print "========================================"
    print "Pos. class accuracy: %s" % (m_tp/float(m_tp+fn),)
    print "Neg. class accuracy: %s" % (m_tn/float(m_tn+m_fp),)
    print "Corpus Precision {:.3}".format(prec)
    print "Corpus Recall    {:.3}".format(rec)
    print "Corpus F1        {:.3}".format(f1)
    print "----------------------------------------"
    print "TP: {} | FP: {} | TN: {} | FN: {}".format(m_tp, m_fp, m_tn, fn)
    print "========================================\n"

def precision(pred, gold):
    tp = np.sum((pred == 1) * (gold == 1))
    fp = np.sum((pred == 1) * (gold != 1))
    return 0 if tp == 0 else float(tp) / float(tp + fp)

def recall(pred, gold):
    tp = np.sum((pred == 1) * (gold == 1))
    p  = np.sum(gold == 1)
    return 0 if tp == 0 else float(tp) / float(p)

def f1_score(pred, gold):
    prec = precision(pred, gold)
    rec  = recall(pred, gold)
    return 0 if (prec * rec == 0) else 2 * (prec * rec)/(prec + rec)

def test_scores(pred, gold, return_vals=True, verbose=False):
    """Returns: (precision, recall, f1_score, tp, fp, tn, fn, n_test)"""
    n_t = len(gold)
    if np.sum(gold == 1) + np.sum(gold == -1) != n_t:
        raise ValueError("Gold labels must be in {-1,1}.")
    tp   = np.sum((pred == 1) * (gold == 1))
    fp   = np.sum((pred == 1) * (gold == -1))
    tn   = np.sum((pred < 1) * (gold == -1))
    fn   = np.sum((pred < 1) * (gold == 1))
    prec = tp / float(tp + fp)
    rec  = tp / float(tp + fn)
    f1   = 2 * (prec * rec) / (prec + rec)

    # Print simple report if verbose=True
    if verbose:
        print "=" * 40
        print "Test set size:\t%s" % n_t
        print "-" * 40
        print "Pos. class accuracy: %s" % (tp/float(tp+fn),)
        print "Neg. class accuracy: %s" % (tn/float(tn+fp),)
        print "-" * 40
        print "Precision:\t%s" % prec
        print "Recall:\t\t%s" % rec
        print "F1 Score:\t%s" % f1
        print "-" * 40
        print "TP: %s | FP: %s | TN: %s | FN: %s" % (tp,fp,tn,fn)
        print "=" * 40
    if return_vals:
        return prec, rec, f1, tp, fp, tn, fn, n_t
    
def scores_from_counts(tp, fp, tn, fn):
    prec = float(len(tp)) / (len(tp) + len(fp)) if len(tp) > 0 else 0
    rec = float(len(tp)) / (len(tp) + len(fn)) if len(tp) > 0 else 0
    f1 = 2.0 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1    

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


ValidatedFit = namedtuple('ValidatedFit', ['w', 'P', 'R', 'F1'])


def grid_search_plot(w_fit, mu_opt, f1_opt):
    """ Plot validation set performance for logistic regression regularization """
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
    ax1.set_position([box1.x0, box1.y0+box1.height*0.1, box1.width, box1.height*0.9])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0+box2.height*0.1, box2.width, box2.height*0.9])
    plt.title("Validation for logistic regression learning")
    lns1, lbs1 = ax1.get_legend_handles_labels()
    lns2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1+lns2, lbs1+lbs2, loc='upper center', bbox_to_anchor=(0.5,-0.05),
               scatterpoints=1, fontsize=10, markerscale=0.5)
    plt.show()

    
class Parameter(object):
    """Base class for a grid search parameter"""
    def __init__(self, name):
        self.name = name
    
    def get_all_values(self):
        raise NotImplementedError()
    
    def draw_values(self, n):
        return np.random.choice(self.get_all_values(), n)
    
class ListParameter(Parameter):
    """List of parameter values for searching"""
    def __init__(self, name, parameter_list):
        self.parameter_list = np.ravel(parameter_list)
        super(ListParameter, self).__init__(name)
    
    def get_all_values(self):
        return self.parameter_list
    
class RangeParameter(Parameter):
    """
    Range of parameter values for searching.
    min_value and max_value are the ends of the search range
    If log_base is specified, scale the search range in the log base
    step is range step size or exponent step size
    """
    def __init__(self, name, min_value, max_value, step=1, log_base=None):
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.log_base = log_base
        super(RangeParameter, self).__init__(name)
        
    def get_all_values(self):
        if self.log_base:
            min_exp = math.log(self.min_value, self.log_base)
            max_exp = math.log(self.max_value, self.log_base)
            exps = np.arange(min_exp, max_exp + self.step, step=self.step)
            return np.power(self.log_base, exps)
        return np.arange(self.min_value, self.max_value + self.step, step=self.step)
        

class GridSearch(object):
    """
    Runs hyperparameter grid search over a model object with train and score methods,
    training data (X), and training_marginals
    Selects based on maximizing F1 score on a supplied validation set
    Specify search space with Parameter arguments
    """
    def __init__(self, model, X, training_marginals, *parameters):
        self.model              = model
        self.X                  = X
        self.training_marginals = training_marginals
        self.params             = parameters
        self.param_names        = [param.name for param in parameters]
        
    def search_space(self):
        return product(param.get_all_values() for param in self.params)

    def fit(self, X_validation, validation_labels, gold_candidate_set, b=0.5, set_unlabeled_as_neg=True, **model_hyperparams):
        """
        Basic method to start grid search, returns DataFrame table of results
          b specifies the positive class threshold for calculating f1
          set_unlabeled_as_neg is used to decide class of unlabeled cases for f1
          Non-search parameters are set using model_hyperparamters
        """
        # Iterate over the param values
        run_stats   = []
        param_opts  = np.zeros(len(self.param_names))
        f1_opt      = -1.0
        for param_vals in self.search_space():

            # Set the new hyperparam configuration to test
            for pn, pv in zip(self.param_names, param_vals):
                model_hyperparams[pn] = pv
            print "=" * 60
            print "Testing %s" % ', '.join(["%s = %0.2e" % (pn,pv) for pn,pv in zip(self.param_names, param_vals)])
            print "=" * 60

            # Train the model
            self.model.train(self.X, self.training_marginals, **model_hyperparams)

            # Test the model
            tp, fp, tn, fn = self.model.score(X_validation, validation_labels, gold_candidate_set, b, set_unlabeled_as_neg, display=False)
            p, r, f1 = scores_from_counts(tp, fp, tn, fn)
            run_stats.append(list(param_vals) + [p, r, f1])
            if f1 > f1_opt:
                w_opt      = self.model.w
                param_opts = param_vals
                f1_opt     = f1

        # Set optimal parameter in the learner model
        self.model.w = w_opt

        # Return DataFrame of scores
        self.results = DataFrame.from_records(run_stats, columns=self.param_names + ['Prec.', 'Rec.', 'F1'])
        return self.results
    
    
class RandomSearch(GridSearch):
    def __init__(self, model, X, training_marginals, n, *parameters):
        """Search a random sample of size n from a parameter grid"""
        self.n = n
        super(RandomSearch, self).__init__(model, X, training_marginals, *parameters)
        
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
        print "=" * 60
        print "LF Summary Statistics: %s LFs applied to %s candidates" % (M, N)
        print "-" * 60
        print "Coverage (candidates w/ > 0 labels):\t\t%0.2f%%" % (coverage*100,)
        print "Overlap (candidates w/ > 1 labels):\t\t%0.2f%%" % (overlap*100,)
        print "Conflict (candidates w/ conflicting labels):\t%0.2f%%" % (conflict*100,)
        print "=" * 60
    if return_vals:
        return coverage, overlap, conflict
