# Base Python
import cPickle, json, os, sys, warnings
from collections import defaultdict, OrderedDict, namedtuple
import lxml.etree as et

# Scientific modules
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


class Learner(object):
    """
    Core learning class for Snorkel, encapsulating the overall pipeline:
        1. Generating a noisy training set by applying the LFs to the candidates
        2. Modeling this noisy training set
        3. Learning a model over the candidate features, trained on the noisy training set

    As input takes:
        - A set of Candidate objects
        - A set of labeling functions (LFs) which are functions f : Candidate -> {-1,0,1}
        - A Featurizer object, which is applied to the Candidate objects to generate features
        - A Model object, representing the model to train
        - _A test set to compute performance against, which consists of a dict mapping candidate id -> T/F_
    """
    # TODO: Tuner (GridSearch) class that wraps this! 
    def __init__(self, candidates, lfs, model=None, featurizer=None, test_set={}, exclude_cids=[]):
        self.lfs = lfs

        # If test_set is provided (as a dict mapping cid -> T/F), split candidates into training/test
        exclude_cids          = list(set(exclude_cids + test_set.keys()))
        self.train_candidates = filter(lambda c : c.id not in exclude_cids, candidates)
        self.test_candidates  = filter(lambda c : c.id in test_set.keys(), candidates)

        # (1) Generate the noisy training set T by applying the LFs
        # T_{i,j} is the label (in {-1,0,1}) given to candidate i by LF j
        print "Applying LFs to Candidates..."
        self.L_train = self._apply_lfs(self.train_candidates)
        self.L_test  = self._apply_lfs(self.test_candidates)

        # Generate features; F_{i,j} is 1 if candidate i has feature j
        if featurizer is not None:
            print "Feauturizing candidates..."
            self.F_train = featurizer.fit_apply(self.train_candidates)
            self.F_test  = featurizer.apply(self.test_candidates)

        # Dimensions
        self.n_train, self.m = self.L_train.shape
        self.f               = self.F_train.shape[1]
        self.n_test          = self.L_test.shape[0]

        # Model and other attributes
        self.model    = model
        self.test_set = test_set
        self.X_train  = None
        self.X_test   = None

    def _apply_lfs(self, candidates):
        """Apply the labeling functions to the candidates to populate X"""
        X = sparse.lil_matrix((len(candidates), len(self.lfs)))
        for i,c in enumerate(candidates):
            for j,lf in enumerate(self.lfs):
                self.X[i,j] = lf(c)
        return X.tocsr()

    def _print_test_score(self):
        predicted = self.model.predict(self.X_test)

        # TODO: Map test_set values -> order of X_test rows!!!

        # TODO: calculate precision and recall
        
        # TODO: Print!
        raise NotImplementedError()

    def train_model(self, feat_w0=0.0, lf_w0=1.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        # TODO: Bias term
        self.X_train = sparse.hstack([self.L_train, self.F_train], format='csc')
        self.X_test  = sparse.hstack([self.L_test, self.F_test], format='csc')

        # Set initial values for feature weights
        w0 = np.concatenate([lf_w0*np.ones(self.m), feat_w0*np.ones(self.f)])

        # Train model
        self.model.train(self.X_train, w0=w0, **model_hyperparams)
        
        # Print out score if test_set was provided
        self._print_test_score()

    def lf_weights(self):
        return self.model.w[:self.m]

    def lf_accs(self):
        return odds_to_prob(self.lf_weights())

    def feat_weights(self):
        return self.model.w[self.m:]
        

class PipelinedLearner(Learner):
    """Implements the **"pipelined" approach**"""
    def train_model(self, feat_w0=0.0, lf_w0=1.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        # Learn lf accuracies first
        self.lf_accs = LogReg(self.L_train, w0=lf_w0*np.ones(self.m), **model_hyperparams)

        # Compute marginal probabilities over the candidates from this model of the training set
        # TODO

        # Learn model over features
        self.w = self.model.train( \
            self.F, training_marginals=self.training_marginals, w0=feat_w0*np.ones(self.f), **model_hyperparams)

        # Print out score if test_set was provided
        self._print_test_score()
