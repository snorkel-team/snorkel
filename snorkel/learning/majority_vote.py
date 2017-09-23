import numpy as np

from classifier import Classifier

class MajorityVoter(Classifier):
    """Simple Classifier that makes the majority vote given an AnnotationMatrix."""

    def marginals(self, X, **kwargs):
        return np.where(np.ravel(np.sum(X, axis=1)) <= 0, 0.0, 1.0)