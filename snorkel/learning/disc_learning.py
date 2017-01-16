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

    def __init__(self, save_file=None, name='TFModel'):
        """Interface for a TensorFlow model
        The train_fn, loss, and prediction fields should
        be populated by build()
        """
        super(TFNoiseAwareModel, self).__init__(name)
        self.train_fn   = None
        self.loss       = None
        self.prediction = None
        self.session    = tf.Session()
        # Load model
        if save_file is not None:
            self.load(save_file)

    def _build(self, **kwargs):
        """Builds the TensorFlow model
        Populates @train_fn, @loss, @prediction
        Returns dictionary of variables to save
        """
        raise NotImplementedError()

    def save_info(self, model_name, **kwargs):
        pass

    def load_info(self, model_name, **kwargs):
        pass

    def save(self, save_dict, model_name=None, verbose=False):
        model_name = model_name or self.name
        self.save_info(model_name)
        saver = tf.train.Saver(save_dict)
        saver.save(self.session, './' + model_name, global_step=0)
        if verbose:
            print("[{0}] Model saved. To load, use name\n\t\t{1}".format(
                self.name, model_name
            ))

    def load(self, model_name, verbose=False):
        self.load_info(model_name)
        load_dict = self._build()
        saver = tf.train.Saver(load_dict)
        ckpt = tf.train.get_checkpoint_state('./')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.session, ckpt.model_checkpoint_path)
            if verbose:
                print("[{0}] Loaded model <{1}>".format(self.name, model_name))
        else:
            raise Exception("[{0}] No model found at <{1}>".format(
                self.name, model_name
            ))


def get_train_idxs(marginals, rebalance=False, split_lo=0.5, split_hi=0.5):
    pos = np.where(marginals < (split_lo - 1e-6))[0]
    neg = np.where(marginals > (split_hi + 1e-6))[0]
    if rebalance:
        k = min(len(pos), len(neg))
        pos = np.random.choice(pos, size=k, replace=False)
        neg = np.random.choice(neg, size=k, replace=False)
    idxs = np.concatenate([pos, neg])
    np.random.shuffle(idxs)
    return idxs
