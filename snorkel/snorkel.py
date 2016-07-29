import cPickle, json, os, sys, warnings
from collections import defaultdict, OrderedDict, namedtuple
import lxml.etree as et
import numpy as np
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from features import Featurizer
from learning import LogReg, odds_to_prob
from lstm import *
from learning_utils import test_scores, calibration_plots, training_set_summary_stats, sparse_abs, LF_coverage, \
    LF_overlaps, LF_conflicts, LF_accuracies
from pandas import Series, DataFrame


class TrainingSet(object):
    """
    Wrapper data object which applies the LFs to the candidates comprising the training set,
    featurizes them, and then stores the resulting _noisy training set_, as well as the LFs and featurizer.

    As input takes:
        - A set of Candidate objects comprising the training set
        - A set of labeling functions (LFs) which are functions f : Candidate -> {-1,0,1}
        - A Featurizer object, which is applied to the Candidate objects to generate features
    """
    def __init__(self, training_candidates, lfs, featurizer=None):
        self.training_candidates = training_candidates
        self.featurizer          = featurizer
        self.lfs                 = lfs
        self.lf_names            = [lf.__name__ for lf in lfs]
        self.L, self.F           = self.transform(self.training_candidates, fit=True)
        self.dev_candidates      = None
        self.dev_labels          = None
        self.L_dev               = None
        self.summary_stats()

    def transform(self, candidates, fit=False):
        """Apply LFs and featurize the candidates"""
        print "Applying LFs..."
        L = self._apply_lfs(candidates)
        F = None
        if self.featurizer is not None:
            print "Featurizing..."
            F = self.featurizer.fit_transform(candidates) if fit else self.featurizer.transform(candidates)
        return L, F

    def _apply_lfs(self, candidates):
        """Apply the labeling functions to the candidates to populate X"""
        X = sparse.lil_matrix((len(candidates), len(self.lfs)))
        for i,c in enumerate(candidates):
            for j,lf in enumerate(self.lfs):
                X[i,j] = lf(c)
        return X.tocsr()

    def summary_stats(self, return_vals=False, verbose=True):
        """Print out basic stats about the LFs wrt the training candidates"""
        return training_set_summary_stats(self.L, return_vals=return_vals, verbose=verbose)

    def lf_stats(self, dev_candidates=None, dev_labels=None):
        """Returns a pandas Dataframe with the LFs and various per-LF statistics"""
        N, M = self.L.shape

        # Default LF stats
        d = {
            'j'         : range(len(self.lfs)),
            'coverage'  : Series(data=LF_coverage(self.L), index=self.lf_names),
            'overlaps'  : Series(data=LF_overlaps(self.L), index=self.lf_names),
            'conflicts' : Series(data=LF_conflicts(self.L), index=self.lf_names)
        }
        
        # Empirical stats, based on supplied development set
        if dev_candidates and dev_labels is not None:
            if self.L_dev is None or dev_candidates != self.dev_candidates or any(dev_labels != self.dev_labels):
                self.dev_candidates = dev_candidates
                self.dev_labels     = dev_labels
                self.L_dev          = self._apply_lfs(dev_candidates)
            d['accuracy'] = Series(data=LF_accuracies(self.L_dev, self.dev_labels), index=self.lf_names)
        return DataFrame(data=d, index=self.lf_names)


class Learner(object):
    """
    Core learning class for Snorkel, encapsulating the overall process of learning a generative model of the
    training data set (specifically: of the LF-emitted labels and the true class labels), and then using this
    to train a given noise-aware discriminative model.

    As input takes a TrainingSet object and a NoiseAwareModel object (the discriminative model to train).
    """
    # TODO: Tuner (GridSearch) class that wraps this! 
    def __init__(self, training_set, model=None):
        self.training_set = training_set
        self.model        = model

        # Derived objects from the training set
        self.L_train         = self.training_set.L
        self.n_train, self.m = self.L_train.shape
        self.F_train         = self.training_set.F
        self.f               = self.F_train.shape[1]
        self.X_train         = None

        # Cache the transformed test set as well
        self.test_candidates = None
        self.gold_labels     = None
        self.X_test          = None

    def _set_model_X(self, L, F):
        """Given LF matrix L, feature matrix F, return the matrix used by the end discriminative model."""
        return sparse.hstack([L, F], format='csc')

    def train(self, lf_w0=5.0, feat_w0=0.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        # TODO: Bias term
        # Set the initial weights for LFs and feats
        w0 = np.concatenate([lf_w0*np.ones(self.m), feat_w0*np.ones(self.f)])

        # Construct matrix X for "joint" approach
        self.X_train = self._set_model_X(self.L_train, self.F_train)

        # Train model
        self.model.train(self.X_train, w0=w0, **model_hyperparams)

    def test(self, test_candidates, gold_labels, display=True, return_vals=False):
        """
        Apply the LFs and featurize the test candidates, using the same transformation as in training set;
        then test against gold labels using trained model.
        """
        # Cache transformed test set
        if self.X_test is None or test_candidates != self.test_candidates or any(gold_labels != self.gold_labels):
            self.test_candidates = test_candidates
            self.gold_labels     = gold_labels
            L_test, F_test       = self.training_set.transform(test_candidates)
            self.X_test          = self._set_model_X(L_test, F_test)
        if display:
            calibration_plots(self.model.marginals(self.X_train), self.model.marginals(self.X_test), gold_labels)
        return test_scores(self.model.predict(self.X_test), gold_labels, return_vals=return_vals, verbose=display)


    def lf_weights(self):
        return self.model.w[:self.m]

    def lf_accs(self):
        return odds_to_prob(self.lf_weights())

    def feat_weights(self):
        return self.model.w[self.m:]
        

class PipelinedLearner(Learner):
    """Implements the **"pipelined" approach**"""
    def _set_model_X(self, L, F):
        return F.tocsc()

    def train(self, feat_w0=0.0, lf_w0=1.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        w0_1 = lf_w0*np.ones(self.m)
        w0_2 = feat_w0*np.ones(self.f)

        # Learn lf accuracies first
        self.training_model = LogReg()
        self.training_model.train(self.L_train, w0=w0_1, **model_hyperparams)

        # Compute marginal probabilities over the candidates from this model of the training set
        training_marginals = self.training_model.marginals(self.L_train)

        # Learn model over features
        self.X_train = self._set_model_X(self.L_train, self.F_train)
        self.w       = self.model.train(self.X_train, training_marginals=training_marginals, w0=w0_2, \
            **model_hyperparams)

    def lf_weights(self):
        return self.training_model.w

    def feat_weights(self):
        return self.model.w
