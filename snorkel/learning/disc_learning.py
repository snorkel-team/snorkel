import tensorflow as tf
import numpy as np
from time import time
import os
from six.moves.cPickle import dump, load

from .classifier import Classifier
from .utils import reshape_marginals, LabelBalancer


class NoiseAwareModel(Classifier):
    """Simple abstract base class for a model."""
    def __init__(self, cardinality=2, name=None, seed=None):
        self.name = name or self.__class__.__name__
        self.cardinality = cardinality
        self.seed = seed

    def train(self, X, training_marginals, **training_kwargs):
        """Trains the model using probabilistic labels (training marginals)."""
        raise NotImplementedError()


class TFNoiseAwareModel(NoiseAwareModel):
    """
    Generic NoiseAwareModel class for TensorFlow models.
    Note that the actual network is built when train is called (to allow for
    model architectures which depend on the training data, e.g. vocab size).
    @n_threads: Parallelism to use; single-threaded if None
    """
    def __init__(self, n_threads=None, **kwargs):
        self.n_threads = n_threads
        super(TFNoiseAwareModel, self).__init__(**kwargs)

    def _build_model(self, **model_kwargs):
        """
        Builds the TensorFlow Operations for the model. Must set the following:
            @self.logits: The un-normalized potentials for the variables
                ("logits" in keeping with TensorFlow terminology)
            @Y: The training marginals to fit to
            @self.marginals_op: Normalized predicted marginals for the variables

        Additional ops must be set depending on whether the default
        self._construct_feed_dict method below is used, or a custom one.

        Note that _build_model is called in the train method, allowing for the 
        network to be constructed dynamically depending on the training set 
        (e.g., the size of the vocabulary for a text LSTM model)

        Note also that model_kwargs are saved to disk by the self.save method,
        as they are needed to rebuild / reload the model. *All hyperparameters
        needed to rebuild the model must be passed in here for model reloading
        to work!*
        """
        raise NotImplementedError()

    def _build_training_ops(self, **training_kwargs):
        """
        Builds the TensorFlow Operations for the training procedure. Must set 
        the following:
            @self.loss: Loss function
            @self.optimizer: Training operation
        """
        # Define loss and marginals ops
        if self.cardinality > 2:
            loss_fn = tf.nn.softmax_cross_entropy_with_logits
        else:
            loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
        self.loss = tf.reduce_sum(loss_fn(logits=self.logits, labels=self.Y))
        
        # Build training op
        self.lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _construct_feed_dict(self, X_b, Y_b, lr=0.01, **kwargs):
        """
        Given a batch of data and labels, and other necessary hyperparams,
        construct a python dictionary to use in the training step as feed_dict.

        This method can be overwritten when non-standard arguments are needed.

        NOTE: Using a feed_dict is obviously not the fastest way to pass in data
            if running a large-scale dataset on a GPU, see Issue #679.
        """
        # Note: The below arguments need to be constructed by self._build!
        return {self.X: X_b, self.Y: Y_b, self.lr: lr}

    def _build_new_graph_session(self, **model_kwargs):
        """Creates new graph, builds network, and sets new session."""
        self.model_kwargs = model_kwargs

        # Create new computation graph
        self.graph = tf.Graph()

        # Build network here in the graph
        with self.graph.as_default():
            self._build_model(**model_kwargs)

        # Create new session
        self.session = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=self.n_threads,
                inter_op_parallelism_threads=self.n_threads
            ),
            graph=self.graph
        ) if self.n_threads is not None else tf.Session(graph=self.graph)

    def _check_input(self, X):
        """Checks correctness of input; optional to implement."""
        pass

    def train(self, X_train, Y_train, n_epochs=25, lr=0.01, batch_size=256, 
        rebalance=False, X_dev=None, Y_dev=None, print_freq=5, **kwargs):
        """
        Generic training procedure for TF model

        @X_train: The training Candidates. If self.representation is True, then
            this is a list of Candidate objects; else is a csr_AnnotationMatrix
            with rows corresponding to training candidates and columns 
            corresponding to features.
        @Y_train: Array of marginal probabilities for each Candidate
        @n_epochs: Number of training epochs
        @lr: Learning rate
        @batch_size: Batch size for SGD
        @rebalance: Bool or fraction of positive examples for training
                    - if True, defaults to standard 0.5 class balance
                    - if False, no class balancing
        @X_dev: Candidates for evaluation, same format as X_train
        @Y_dev: Labels for evaluation, same format as Y_train
        @print_freq: number of epochs at which to print status
        @kwargs: All hyperparameters that change how the graph is built 
            must be passed through here to be saved and reloaded to save /
            reload model. *NOTE: If a parameter needed to build the 
            network and/or is needed at test time is not included here, the
            model will not be able to be reloaded!*
        """
        self._check_input(X_train)
        np.random.seed(self.seed)
        verbose = print_freq > 0

        # If the data passed in is a feature matrix (representation=False),
        # set the dimensionality here; else assume this is done by sub-class
        if not self.representation:
            self.d = X_train.shape[1]

        # Check that the cardinality of the training marginals and model agree
        cardinality = Y_train.shape[1] if len(Y_train.shape) > 1 else 2
        if cardinality != self.cardinality:
            raise ValueError("Training marginals cardinality ({0}) does not"
                "match model cardinality ({1}).".format(Y_train.shape[1], 
                    self.cardinality))
        # Make sure marginals are in correct default format
        Y_train = reshape_marginals(Y_train)
        # Make sure marginals are in [0,1] (v.s e.g. [-1, 1])
        if self.cardinality > 2 and not np.all(Y_train.sum(axis=1) - 1 < 1e-10):
            raise ValueError("Y_train must be row-stochastic (rows sum to 1).")
        if not np.all(Y_train >= 0):
            raise ValueError("Y_train must have values in [0,1].")

        # Rebalance training set (only for binary setting currently)
        if self.cardinality == 2:
            train_idxs = LabelBalancer(Y_train).get_train_idxs(rebalance)
            X_train = [X_train[j] for j in train_idxs] if self.representation \
                else X_train[train_idxs, :]
            Y_train = np.ravel(Y_train)[train_idxs]

        # Create new graph, build network, and start session
        self._build_new_graph_session(**kwargs)

        # Build training ops
        # Note that training_kwargs and model_kwargs are mized together; ideally
        # would be separated but no negative effect
        with self.graph.as_default():
            self._build_training_ops(**kwargs)

        # Process the dev set if provided
        if X_dev is not None and Y_dev is not None:
            Y_dev = np.ravel(Y_dev) if self.cardinality == 2 else Y_dev
        
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
                    X_train[i:min(n, i+batch_size)],
                    Y_train[i:min(n, i+batch_size)],
                    lr=lr,
                    **kwargs
                )
                # Run training step and evaluate loss function    
                epoch_loss, _ = self.session.run(
                    [self.loss, self.optimizer], feed_dict=feed_dict)
                epoch_losses.append(epoch_loss)
            
            # Print training stats
            if verbose and (t % print_freq == 0 or t in [0, (n_epochs-1)]):
                msg = "[{0}] Epoch {1} ({2:.2f}s)\tAverage loss={3:.6f}".format(
                    self.name, t, time() - st, np.mean(epoch_losses))
                if X_dev is not None:
                    scores = self.score(X_dev, Y_dev)
                    score = scores if self.cardinality > 2 else scores[-1]
                    score_label = "Acc." if self.cardinality > 2 else "F1"
                    msg += '\tDev {0}={1:.2f}'.format(score_label, 100. * score)
                print(msg)
        
        # Conclude training
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))

    def save(self, model_name=None, save_dir='checkpoints', verbose=True):
        """Save current model."""
        model_name = model_name or self.name

        # Note: Model checkpoints need to be saved in separate directories!
        model_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Create Saver
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())

        # Save model kwargs needed to rebuild model
        with open(os.path.join(model_dir, "model_kwargs.pkl"), 'wb') as f:
            dump(self.model_kwargs, f)

        # Save graph and report if verbose
        saver.save(self.session, os.path.join(model_dir, model_name))
        if verbose:
            print("[{0}] Model saved as <{1}>".format(self.name, model_name))

    def load(self, model_name, save_dir='checkpoints', verbose=True):
        """Load model from file and rebuild in new graph / session."""
        model_dir = os.path.join(save_dir, model_name)

        # Load model kwargs needed to rebuild model
        with open(os.path.join(model_dir, "model_kwargs.pkl"), 'rb') as f:
            model_kwargs = load(f)
        
        # Create new graph, build network, and start session
        self._build_new_graph_session(**model_kwargs)

        # Initialize variables
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())

        # Load saved checkpoint to populate trained parameters
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.session, ckpt.model_checkpoint_path)
            if verbose:
                print("[{0}] Loaded model <{1}>".format(self.name, model_name))
        else:
            raise Exception("[{0}] No model found at <{1}>".format(
                self.name, model_name))
