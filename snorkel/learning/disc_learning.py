import tensorflow as tf
import numpy as np
from time import time
import os
from six.moves.cPickle import dump, load

from sqlalchemy.sql import bindparam, select

from .utils import MentionScorer, reshape_marginals
from ..annotations import save_marginals
from ..models import Candidate


class NoiseAwareModel(object):
    """Simple abstract base class for a model."""

    # Set this class variable to True if train, marginals, predict, and score,
    # take a list of @Candidates as the first argument X;
    # otherwise assume X is an AnnotationMatrix
    representation = False

    def __init__(self, cardinality=2, name=None):
        self.name = name or self.__class__.__name__
        self.cardinality = cardinality

    def train(self, X, training_marginals, **training_kwargs):
        """Trains the model."""
        raise NotImplementedError()

    def marginals(self, X, **kwargs):
        raise NotImplementedError()

    def save_marginals(self, session, X):
        """Save the predicted marginal probabilitiess for the Candidates X."""
        save_marginals(session, X, self.marginals(X), training=False)

    def predictions(self, X, b=0.5):
        """Return numpy array of elements in {-1,0,1}
        based on predicted marginal probabilities.
        """
        if self.cardinality > 2:
            return self.marginals(X).argmax(axis=0) + 1
        else:
            return np.array([1 if p > b else -1 if p < b else 0 
                for p in self.marginals(X)])

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
    """
    Generic NoiseAwareModel class for TensorFlow models.
    Note that the actual network is built when train is called (to allow for
    model architectures which depend on the training data, e.g. vocab size).
    @n_threads: Parallelism to use; single-threaded if None
    """
    # TODO: Pass on model kwargs in GridSearch!
    # TODO: Clean up scoring function in general!
    # TODO: Test + update LogisticRegression (move this to contrib/end_models)?
    def __init__(self, n_threads=None, **kwargs):
        self.n_threads = n_threads
        super(TFNoiseAwareModel, self).__init__(**kwargs)

    def _build(self, **model_kwargs):
        """
        Builds the TensorFlow Operations for the model and training procedure.
        Must set the following ops:
            @self.loss_op: Loss function
            @self.train_op: Training operation
            @self.prediction_op: Prediction of the network

        Additional ops must be set depending on whether the default
        self._construct_feed_dict method below is used, or a custom one.

        Note that _build is called in the train method, allowing for the network
        to be constructed dynamically depending on the training set (e.g., the
        size of the vocabulary for a text LSTM model)
        """
        raise NotImplementedError()

    def _construct_feed_dict(self, X_b, Y_b, lr=0.01, dropout=None, **kwargs):
        """
        Given a batch of data and labels, and other necessary hyperparams,
        construct a python dictionary to use in the training step as feed_dict.

        This method can be overwritten when non-standard arguments are needed.

        NOTE: Using a feed_dict is obviously not the fastest way to pass in data
            if running a large-scale dataset on a GPU, see Issue #679.
        """
        # Note: The below arguments need to be constructed by self._build!
        return {
            self.X: X_b,
            self.Y: Y_b,
            self.keep_prob: dropout or 1.0,
            self.lr: lr
        }

    def _build_session(self, **model_kwargs):
        """Creates new graph, builds network, and sets new session."""
        self.model_kwargs = model_kwargs

        # Create new computation graph
        self.graph = tf.Graph()

        # Build network here in the graph
        with self.graph.as_default():
            self._build(**model_kwargs)

        # Create new session
        self.session = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=self.n_threads,
                inter_op_parallelism_threads=self.n_threads
            ),
            graph=self.graph
        ) if self.n_threads is not None else tf.Session(graph=self.graph)

    def train(self, X_train, Y_train, n_epochs=25, lr=0.01, dropout=0.5, 
        batch_size=256, rebalance=False, X_dev=None, Y_dev=None, print_freq=5, 
        scorer=MentionScorer, **model_kwargs):
        """
        Generic training procedure for TF model

        @X_train: The training Candidates. If self.representation is True, then
            this is a list of Candidate objects; else is a csr_AnnotationMatrix
            with rows corresponding to training candidates and columns 
            corresponding to features.
        @Y_train: Array of marginal probabilities for each Candidate
        @n_epochs: Number of training epochs
        @lr: Learning rate
        @dropout: Keep probability for dropout layer (no dropout if None)
        @batch_size: Batch size for SGD
        @rebalance: Bool or fraction of positive examples for training
                    - if True, defaults to standard 0.5 class balance
                    - if False, no class balancing
        @X_dev: Candidates for evaluation, same format as X_train
        @Y_dev: Labels for evaluation, same format as Y_train
        @print_freq: number of epochs at which to print status
        @scorer: Scorer class to use for dev set evaluations (if provided)
        @model_kwargs: Model hyperparameters that change how the graph is built;
            these must be saved and re-used to re-load model (vs. other keyword
            args in train, which only affect how the model is trained).
        """
        np.random.seed(self.seed)
        verbose = print_freq > 0

        # Check that the cardinality of the training marginals and model agree
        if Y_train.shape[1] != self.cardinality:
            raise ValueError("Training marginals cardinality ({0}) does not"
                "match model cardinality ({1}).".format(Y_train.shape[1], 
                    self.cardinality))
        # Make sure marginals are in correct default format
        Y_train = reshape_marginals(Y_train)

        # Rebalance training set (only for binary setting currently)
        if self.cardinality == 2:
            train_idxs = LabelBalancer(Y_train).get_train_idxs(rebalance)
            X_train = [X_train[j] for j in train_idxs] if self.representation \
                else X_train[train_idxs,]
            Y_train = np.ravel(Y_train)[train_idxs]

        # Create new graph, build network, and start session
        self._build_session(**model_kwargs)

        # Process the dev set if provided
        if X_dev is not None and Y_dev is not None:
            Y_dev = np.ravel(Y_dev) if self.cardinality == 2 else Y_dev
            if not ((Y_dev >= 0).all() and (Y_dev <= 1).all()):
                raise Exception("Y_dev elements should be in [0, 1]")
            dev_scorer = scorer(X_dev, Y_dev)
        
        # Initialize variables
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())

        # Run mini-batch SGD
        n = len(X_train) if self.representation else X_train.shape[0]
        batch_size = min(batch_size, n)
        if verbose:
            st = time()
            print("[{0}] Training model".format(self.name))
            print("[{0}] n_train={1}  #epochs={2}  batch size={3}".format(
                self.name, n, n_epochs, batch_size
            ))
        for t in range(n_epochs):
            epoch_losses = []
            for i in range(0, n, batch_size):
                feed_dict = self._construct_feed_dict(
                    X_train[i:i+batch_size],
                    Y_train[i:i+batch_size],
                    lr=lr,
                    dropout=dropout
                )
                # Run training step and evaluate loss function    
                epoch_loss, _ = self.session.run(
                    [self.loss_op, self.train_op], feed_dict=feed_dict)
                epoch_losses.append(epoch_loss)
            
            # Print training stats
            if verbose and (t % print_freq == 0 or t in [0, (n_epochs-1)]):
                msg = "[{0}] Epoch {1} ({2:.2f}s)\tAverage loss={3:.6f}".format(
                    self.name, t, time() - st, np.mean(epoch_losses))
                if X_dev is not None:
                    score, score_label = dev_scorer.summary_score(
                        self._marginals_preprocessed(X_dev))
                    msg += '\tDev {0}={1:.2f}'.format(score_label, 100. * score)
                print(msg)
        
        # Conclude training
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))

    def save(self, model_name=None, save_dir='./', verbose=True):
        """Save current model."""
        model_name = model_name or self.name

        # Create Saver
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())

        # Save model kwargs needed to rebuild model
        mk_path = os.path.join(save_dir, "{0}.model_kwargs".format(model_name))
        with open(mk_path, 'wb') as f:
            dump(self.model_kwargs, f)

        # Save graph and report if verbose
        saver.save(self.session, os.path.join(save_dir, model_name))
        if verbose:
            print("[{0}] Model saved as <{1}>".format(self.name, model_name))

    def load(self, model_name, save_dir='./', verbose=True):
        """Load model from file and rebuild in new graph / session."""
        # Load model kwargs needed to rebuild model
        mk_path = os.path.join(save_dir, "{0}.model_kwargs".format(model_name))
        with open(mk_path, 'rb') as f:
            model_kwargs = load(f)
        
        # Create new graph, build network, and start session
        self._build_session(**model_kwargs)

        # Load saved checkpoint to populate trained parameters
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.session, ckpt.model_checkpoint_path)
            if verbose:
                print("[{0}] Loaded model <{1}>".format(self.name, model_name))
        else:
            raise Exception("[{0}] No model found at <{1}>".format(
                self.name, model_name))
