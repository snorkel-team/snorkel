import numpy as np

from classifier import Classifier

class MajorityVoter(Classifier):
    """Simple Classifier that makes the majority vote given an AnnotationMatrix."""

    # Set this class variable to True if train, marginals, predict, and score,
    # take a list of @Candidates as the first argument X;
    # otherwise assume X is an AnnotationMatrix
    representation = False

    def marginals(self, X, **kwargs):
        return np.where(np.ravel(np.sum(X, axis=1)) <= 0, 0.0, 1.0)