# Base Python
import cPickle, json, os, sys, warnings
from collections import defaultdict, OrderedDict, namedtuple
import lxml.etree as et

# Scientific modules
import numpy as np
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", module="matplotlib")
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

    # TODO: Take in new (pre-ORM) Candidates here!

    def __init__(self, candidates, lfs, model=LogReg(), featurizer=None, test_set=None):
        
        # (1) Generate the noisy training set T by applying the LFs
        print "Applying LFs to Candidates..."
        self.candidates = candidates
        self.lfs        = lfs

        # T_{i,j} is the label (in {-1,0,1}) given to candidate i by LF j
        self.T = self._apply_lfs(self)

        # Generate features; F_{i,j} is 1 if candidate i has feature j
        if featurizer is not None:
            print "Feauturizing candidates..."
            self.F = featurizer.apply(self.candidates)
        self.model    = model
        self.test_set = test_set
        self.X        = None
        self.w0       = None
        self.w        = None

    def _apply_lfs(self):
        """Apply the labeling functions to the candidates to populate X"""
        # TODO: Parallelize this
        self.X = sparse.lil_matrix((len(self.candidates), len(lfs)))
        for i,c in enumerate(self.candidates):
            for j,lf in enumerate(self.lfs):
                self.X[i,j] = lf(c)
        self.X = self.X.tocsr()

    def _print_test_score():
        # TODO
        raise NotImplementedError()

    def train_model(self, feat_w0=0.0, lf_w0=1.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        # TODO: Bias term
        n, m = self.T.shape
        _, f = self.F.shape
        if self.X is None:
            self.X  = sparse.hstack([self.T, self.F], format='csc')

        # Set initial values for feature weights
        self.w0 = np.concatenate([lf_w0*np.ones(m), feat_w0*np.ones(f)])

        # Train model
        self.w = self.model.train(self.X, w0=self.w0, **model_hyperparams)
        
        # Print out score if test_set was provided
        self._print_test_score()


class PipelinedLearner(Learner):
    """Implements the **"pipelined" approach**"""
    def train_model(self, feat_w0=0.0, lf_w0=1.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        # TODO: Bias term
        m, n = self.T.shape
        f, _ = self.F.shape

        # Learn lf accuracies first
        # TODO: Distinguish between model hyperparams...
        self.lf_accs = LogReg(self.T, w0=lf_w0*np.ones(m), **model_hyperparams)

        # Compute marginal probabilities over the candidates from this model of the training set
        # TODO

        # Learn model over features
        self.w = self.model.train( \
            self.F, training_marginals=self.training_marginals, w0=feat_w0*np.ones(f), **model_hyperparams)

        # Print out score if test_set was provided
        self._print_test_score()
