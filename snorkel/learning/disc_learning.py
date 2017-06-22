import tensorflow as tf
import numpy as np
from time import time

from sqlalchemy.sql import bindparam, select

from .utils import marginals_to_labels, MentionScorer, reshape_marginals
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
    """Generic NoiseAwareModel class for TensorFlow models."""
    def _make_tensor(self, X, **kwargs):
        return X

    def _preprocess_data(self, X, **kwargs):
        return X

    def train(self, X_train, Y_train, n_epochs=25, lr=0.01, dropout=0.5, 
        batch_size=256, rebalance=False, dev_candidates=None, dev_labels=None, 
        print_freq=5, n_threads=None, **model_kwargs):
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
        @rebalance: Bool or fraction of positive examples for training
                    - if True, defaults to standard 0.5 class balance
                    - if False, no class balancing
        @dev_candidates: list of Candidate objects for evaluation
        @dev_labels: array of labels for each dev Candidate
        @print_freq: number of epochs after which to print status
        """
        # TODO: Change dev_* -> X_dev
        # TOOD: Clean up _make_tensor, _preprocess_data
        # TODO: train_fn -> train_op, clean up names and add to build docstring
        # TODO: Generic run_args constructor!
        # TODO: Clean up data preprocessing / train call in RNNBase?
        # TODO: Save model_params (and training_params?)... e.g. rewrite save_info
        # TODO: Make session / graph building a sub-class
        # TODO: Actually save / load model_kwargs to / from disk!
        # TODO: Pass on model kwargs in GridSearch!
        np.random.seed(self.seed)
        verbose = print_freq > 0

        # Check that the cardinality of the training marginals and model agree
        if Y_train.shape[1] != self.cardinality:
            raise ValueError("Training marginals cardinality ({0}) does not"
                "match model cardinality ({1}).".format(Y_train.shape[1], 
                    self.cardinality))
        # Make sure marginals are in correct default format
        Y_train = reshape_marginals(Y_train)

        # Create new computation graph
        self.graph = tf.Graph()
        
        # Get training indices
        # Note: Currently we only do label balancing for binary setting
        if self.cardinality == 2:
            train_idxs = LabelBalancer(Y_train).get_train_idxs(rebalance)
            X_train = [X_train[j] for j in train_idxs] if self.representation \
                else X_train[train_idxs,]
            Y_train = np.ravel(Y_train)[train_idxs]
        
        # Build network here in the graph
        with self.graph.as_default():
            self._build(**model_kwargs)

        # Create new session
        self.session = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=n_threads,
                inter_op_parallelism_threads=n_threads
            ),
            graph=self.graph
        ) if n_threads is not None else tf.Session(graph=self.graph)

        # Get dev data
        dev_data, dev_gold = None, None
        if dev_candidates is not None and dev_labels is not None:
            dev_data, _ = self._preprocess_data(dev_candidates, extend=False)
            dev_gold = np.ravel(dev_labels)
            if not ((dev_gold >= 0).all() and (dev_gold <= 1).all()):
                raise Exception("Dev labels should be in [0, 1]")
            print("[{0}] Loaded {1} candidates for evaluation".format(
                self.name, len(dev_data)
            ))
        
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
            epoch_loss = []
            for i in range(0, n, batch_size):
                # Get batch tensors
                # TODO: Put this in run_ops constructor!
                X_b, len_b = self._make_tensor(X_train[i:i+batch_size])
                Y_b        = Y_train[i:i+batch_size]
                # Run training step and evaluate loss function                  
                epoch_loss.append(self.session.run([self.loss, self.train_fn], {
                    self.sentences:        X_b,
                    self.sentence_lengths: len_b,
                    self.train_marginals:  Y_b,
                    self.keep_prob:        dropout or 1.0,
                    self.lr:               lr
                })[0])
            # Print training stats
            if verbose and (t % print_freq == 0 or t in [0, (n_epochs-1)]):
                msg = "[{0}] Epoch {1} ({2:.2f}s)\tAverage loss={3:.6f}".format(
                    self.name, t, time() - st, np.mean(epoch_loss)
                )
                if dev_data is not None:
                    dev_p    = self._marginals_preprocessed(dev_data)
                    f1, _, _ = f1_score(dev_p, dev_gold)
                    msg     += '\tDev F1={0:.2f}'.format(100. * f1)
                print(msg)
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))

    def _build(self, **model_kwargs):
        """Builds the TensorFlow network"""
        raise NotImplementedError()

    def save_info(self, model_name, **kwargs):
        pass

    def load_info(self, model_name, **kwargs):
        pass

    def save(self, model_name=None, save_file=None, verbose=True, 
        save_dict=None):
        """Save current TensorFlow model
            @model_name: save file names
            @verbose: be talkative?
        """
        model_name = model_name or self.name
        self.save_info(model_name)
        with self.graph.as_default():
            save_dict = save_dict or tf.global_variables()
            saver = tf.train.Saver(save_dict)
        saver.save(self.session, './' + model_name, global_step=0)
        if verbose:
            print("[{0}] Model saved. To load, use name\n\t\t{1}".format(
                self.name, model_name
            ))

    def load(self, model_name, model_kwargs={}, save_file=None, verbose=True, 
        save_dict=None, n_threads=None):
        """Load TensorFlow model from file
            @model_name: save file names
            @verbose: be talkative?
        """
        self.load_info(model_name)
        
        # Create new computation graph
        self.graph = tf.Graph()
        # Build network here in the graph
        with self.graph.as_default():
            self._build(**model_kwargs)
        # Create new session
        self.session = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=n_threads,
                inter_op_parallelism_threads=n_threads
            ),
            graph=self.graph
        ) if n_threads is not None else tf.Session(graph=self.graph)

        # Load saved checkpoint
        with self.graph.as_default():
            load_dict = save_dict or tf.global_variables()
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
