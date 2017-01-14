import numpy as np
import os
import tensorflow as tf

from .utils import marginals_to_labels, MentionScorer


class NoiseAwareModel(object):
    """Simple abstract base class for a model."""

    def __init__(self, name):
        self.name = name
        super(NoiseAwareModel, self).__init__()

    def train(self, X, training_marginals, **hyperparams):
        """Trains the model"""
        raise NotImplementedError()

    def marginals(self, X, **kwargs):
        raise NotImplementedError()

    def predict(self, X, b=0.5):
        """Return numpy array of elements in {-1,0,1} based on predicted marginal probabilities."""
        return marginals_to_labels(self.marginals(X), b)

    def score(self, session, X_test, test_labels, gold_candidate_set=None, b=0.5, set_unlabeled_as_neg=True,
              display=True, scorer=MentionScorer, **kwargs):
        
        # Get the test candidates
        test_candidates = [X_test.get_candidate(session, i) for i in xrange(X_test.shape[0])]

        # Initialize scorer
        s               = scorer(test_candidates, test_labels, gold_candidate_set)
        test_marginals  = self.marginals(X_test, **kwargs)
        train_marginals = (self.marginals(self.X_train) if hasattr(self, 'X_train')
                           and self.X_train is not None else None)
        return s.score(test_marginals, train_marginals, b=b,
                       set_unlabeled_as_neg=set_unlabeled_as_neg, display=display)

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()


class TFNoiseAwareModel(NoiseAwareModel):

    def __init__(self, name='TFModel'):
        """Interface for a TensorFlow model
        The train_fn, loss, and prediction fields should
        be populated by build()
        """
        super(TFNoiseAwareModel, self).__init__(name)
        self.train_fn   = None
        self.loss       = None
        self.prediction = None
        self.session    = tf.Session()

    def _build(self, **kwargs):
        """Builds the TensorFlow model
        Returns a triple of a training function, loss variable, 
        and prediction function
        """
        raise NotImplementedError()

    def save_info(self, model_name, **kwargs):
        pass

    def load_info(self, model_name, **kwargs):
        pass

    def save(self, model_name=None, **kwargs):
        model_name = model_name or self.name
        self.save_info(model_name, **kwargs)
        saver = tf.train.Saver()
        saver.save(self.session, model_name)
        if kwargs.get('verbose', False):
             print("[{0}] Model saved. To load, use name\n\t\t{1}".format(
                self.name, model_name
            ))

    def load(self, model_name, **kwargs):
        self.load_info(model_name, **kwargs)
        self._build(**kwargs)
        saver = tf.train.Saver()
        saver.restore(self.session, '{0}.meta'.format(model_name))
        if kwargs.get('verbose', False):
            print("[{0}] Loaded model <{1}>".format(self.name, model_name))
