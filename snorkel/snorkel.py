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
    def __init__(self, train_candidates, lfs, featurizer=None, model=None):
        self.lfs              = lfs
        self.model            = model
        self.featurizer       = None
        self.train_candidates = train_candidates
        self.train_cids       = [c.id for c in self.train_candidates]
        self.X_train          = None

        # Apply LFs and featurize the candidates
        print "Applying LFs..."
        self.L_train         = self._apply_lfs(self.train_candidates)
        self.n_train, self.m = self.L_train.shape
        print "Featurizing..."
        self.F_train         = featurizer.fit_apply(self.train_candidates) if featurizer else None
        self.f               = self.F_train.shape[1] if self.F_train is not None else 0

    def _apply_lfs(self, candidates):
        """Apply the labeling functions to the candidates to populate X"""
        X = sparse.lil_matrix((len(candidates), len(self.lfs)))
        for i,c in enumerate(candidates):
            for j,lf in enumerate(self.lfs):
                X[i,j] = lf(c)
        return X.tocsr()

    def test(self, test_candidates, gold_labels):
        """Apply the LFs and featurize the test candidates, then test against gold labels using trained model"""
        L_test = self._apply_lfs(test_candidates)
        F_test = featurizer.apply(test_candidates) if featurizer else None
        X_test = sparse.hstack([L_test, F_test], format='csc')
        test_scores(self.model.predict(X_test), gold_labels, return_vals=False, verbose=True)

    def train(self, feat_w0=0.0, lf_w0=1.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        # TODO: Bias term
        self.X_train = sparse.hstack([self.L_train, self.F_train], format='csc')

        # Set initial values for feature weights
        w0 = np.concatenate([lf_w0*np.ones(self.m), feat_w0*np.ones(self.f)])

        # Train model
        self.model.train(self.X_train, w0=w0, **model_hyperparams)

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
