from collections import namedtuple
from itertools import product
import math
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from .utils import scores_from_counts

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

    def fit(self, X_validation, validation_labels, gold_candidate_set=None, b=0.5, set_unlabeled_as_neg=True, **model_hyperparams):
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
