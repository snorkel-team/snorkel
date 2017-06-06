import tensorflow as tf

from sqlalchemy.sql import bindparam, select

from .utils import marginals_to_labels, MentionScorer
from ..annotations import save_marginals
from ..models import Candidate


class NoiseAwareModel(object):
    """Simple abstract base class for a model."""

    # Set this class variable to True if train, marginals, predict, and score,
    # take a list of @Candidates as the first argument X;
    # otherwise assume X is an AnnotationMatrix
    representation = False

    def __init__(self, name):
        self.name = name

    def train(self, X, training_marginals, **hyperparams):
        """Trains the model"""
        raise NotImplementedError()

    def marginals(self, X, **kwargs):
        raise NotImplementedError()

    def save_marginals(self, session, X):
        """Save the predicted marginal probabilitiess for the Candidates X."""
        save_marginals(session, X, self.marginals(X), training=False)

    def predict(self, X, b=0.5):
        """Return numpy array of elements in {-1,0,1}
        based on predicted marginal probabilities.
        """
        return marginals_to_labels(self.marginals(X), b)

    def score(self, session, X_test, test_labels, gold_candidate_set=None, 
        b=0.5, set_unlabeled_as_neg=True, display=True, scorer=MentionScorer,
        **kwargs):
        # Compute the marginals
        test_marginals = self.marginals(X_test, **kwargs)

        # Get the test candidates
        test_candidates = [
            X_test.get_candidate(session, i) for i in xrange(X_test.shape[0])
        ] if not self.representation else X_test

        # Initialize and return scorer
        s = scorer(test_candidates, test_labels, gold_candidate_set)          
        return s.score(test_marginals, train_marginals=None, b=b,
            display=display, set_unlabeled_as_neg=set_unlabeled_as_neg)

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()


class TFNoiseAwareModel(NoiseAwareModel):

    def __init__(self, save_file=None, name='TFModel', n_threads=None):
        """Interface for a TensorFlow model
        The @train_fn, @loss, @prediction, and @save_dict
        fields should be populated by @_build()
        """
        super(TFNoiseAwareModel, self).__init__(name)
        self.train_fn   = None
        self.loss       = None
        self.prediction = None
        self.save_dict  = None
        self.session    = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=n_threads,
                inter_op_parallelism_threads=n_threads
            )
        ) if n_threads is not None else tf.Session()
        # Load model
        if save_file is not None:
            self.load(save_file)

    def _build(self, **kwargs):
        """Builds the TensorFlow model
        Populates @train_fn, @loss, @prediction, @save_dict
        """
        raise NotImplementedError()

    def save_info(self, model_name, **kwargs):
        pass

    def load_info(self, model_name, **kwargs):
        pass

    def save(self, model_name=None, verbose=True):
        """Save current TensorFlow model
            @model_name: save file names
            @verbose: be talkative?
        """
        model_name = model_name or self.name
        self.save_info(model_name)
        save_dict = self.save_dict or tf.global_variables()
        saver = tf.train.Saver(save_dict)
        saver.save(self.session, './' + model_name, global_step=0)
        if verbose:
            print("[{0}] Model saved. To load, use name\n\t\t{1}".format(
                self.name, model_name
            ))

    def load(self, model_name, verbose=True):
        """Load TensorFlow model from file
            @model_name: save file names
            @verbose: be talkative?
        """
        self.load_info(model_name)
        self._build()
        load_dict = self.save_dict or tf.global_variables()
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
