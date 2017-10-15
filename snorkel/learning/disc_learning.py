import tensorflow as tf
import numpy as np
from time import time
import os
from six.moves.cPickle import dump, load

from .classifier import Classifier
from .utils import reshape_marginals, LabelBalancer

class TFNoiseAwareModel(Classifier):
    """
    Generic NoiseAwareModel class for TensorFlow models.
    Note that the actual network is built when train is called (to allow for
    model architectures which depend on the training data, e.g. vocab size).
    
    :param n_threads: Parallelism to use; single-threaded if None
    :param seed: Top level seed which is passed into both numpy operations
        via a RandomState maintained by the class, and into TF as a graph-level
        seed.
    """
    def __init__(self, n_threads=None, seed=123, **kwargs):
        self.n_threads = n_threads
        self.seed = seed
        self.rand_state = np.random.RandomState()
        super(TFNoiseAwareModel, self).__init__(**kwargs)

    def _build_model(self, **model_kwargs):
        """
        Builds the TensorFlow Operations for the model. Must set the following:
            - self.logits: The un-normalized potentials for the variables
                ("logits" in keeping with TensorFlow terminology)
            - self.Y: The training marginals to fit to
            - self.marginals_op: Normalized predicted marginals for the vars

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
            - self.loss: Loss function
            - self.optimizer: Training operation
        """
        # Define loss and marginals ops
        if self.cardinality > 2:
            loss_fn = tf.nn.softmax_cross_entropy_with_logits
        else:
            loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
        self.loss = tf.reduce_mean(loss_fn(logits=self.logits, labels=self.Y))
        
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

        with self.graph.as_default():

            # Set graph-level random seed
            tf.set_random_seed(self.seed)

            # Build network here in the graph
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
        rebalance=False, X_dev=None, Y_dev=None, print_freq=5, dev_ckpt=True,
        dev_ckpt_delay=0.75, save_dir='checkpoints', **kwargs):
        """
        Generic training procedure for TF model

        :param X_train: The training Candidates. If self.representation is True, then
            this is a list of Candidate objects; else is a csr_AnnotationMatrix
            with rows corresponding to training candidates and columns 
            corresponding to features.
        :param Y_train: Array of marginal probabilities for each Candidate
        :param n_epochs: Number of training epochs
        :param lr: Learning rate
        :param batch_size: Batch size for SGD
        :param rebalance: Bool or fraction of positive examples for training
                    - if True, defaults to standard 0.5 class balance
                    - if False, no class balancing
        :param X_dev: Candidates for evaluation, same format as X_train
        :param Y_dev: Labels for evaluation, same format as Y_train
        :param print_freq: number of epochs at which to print status, and if present,
            evaluate the dev set (X_dev, Y_dev).
        :param dev_ckpt: If True, save a checkpoint whenever highest score
            on (X_dev, Y_dev) reached. Note: currently only evaluates at
            every @print_freq epochs.
        :param dev_ckpt_delay: Start dev checkpointing after this portion
            of n_epochs.
        :param save_dir: Save dir path for checkpointing.
        :param kwargs: All hyperparameters that change how the graph is built 
            must be passed through here to be saved and reloaded to save /
            reload model. *NOTE: If a parameter needed to build the 
            network and/or is needed at test time is not included here, the
            model will not be able to be reloaded!*
        """
        self._check_input(X_train)
        verbose = print_freq > 0

        # Set random seed for all numpy operations
        self.rand_state.seed(self.seed)

        # If the data passed in is a feature matrix (representation=False),
        # set the dimensionality here; else assume this is done by sub-class
        if not self.representation:
            kwargs['d'] = X_train.shape[1]

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

        # Remove unlabeled examples (i.e. P(X=k) == 1 / cardinality for all k)
        # and optionally rebalance training set
        # Note: rebalancing only for binary setting currently
        if self.cardinality == 2:
            # This removes unlabeled examples and optionally rebalances
            train_idxs = LabelBalancer(Y_train).get_train_idxs(rebalance,
                rand_state=self.rand_state)
        else:
            # In categorical setting, just remove unlabeled
            diffs = Y_train.max(axis=1) - Y_train.min(axis=1)
            train_idxs = np.where(diffs > 1e-6)[0]
        X_train = [X_train[j] for j in train_idxs] if self.representation \
            else X_train[train_idxs, :]
        Y_train = Y_train[train_idxs]

        # Create new graph, build network, and start session
        self._build_new_graph_session(**kwargs)

        # Build training ops
        # Note that training_kwargs and model_kwargs are mixed together; ideally
        # would be separated but no negative effect
        with self.graph.as_default():
            self._build_training_ops(**kwargs)
        
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
        dev_score_opt = 0.0
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

            # Reshuffle training data
            train_idxs = range(n)
            self.rand_state.shuffle(train_idxs)
            X_train = [X_train[j] for j in train_idxs] if self.representation \
                else X_train[train_idxs, :]
            Y_train = Y_train[train_idxs]
            
            # Print training stats and optionally checkpoint model
            if verbose and (t % print_freq == 0 or t in [0, (n_epochs-1)]):
                msg = "[{0}] Epoch {1} ({2:.2f}s)\tAverage loss={3:.6f}".format(
                    self.name, t, time() - st, np.mean(epoch_losses))
                if X_dev is not None:
                    scores = self.score(X_dev, Y_dev, batch_size=batch_size)
                    score = scores if self.cardinality > 2 else scores[-1]
                    score_label = "Acc." if self.cardinality > 2 else "F1"
                    msg += '\tDev {0}={1:.2f}'.format(score_label, 100. * score)
                print(msg)
                    
                # If best score on dev set so far and dev checkpointing is
                # active, save checkpoint
                if X_dev is not None and dev_ckpt and \
                    t > dev_ckpt_delay * n_epochs and score > dev_score_opt:
                    dev_score_opt = score
                    self.save(save_dir=save_dir, global_step=t)
        
        # Conclude training
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))

        # If checkpointing on, load last checkpoint (i.e. best on dev set)
        if dev_ckpt and X_dev is not None and verbose and dev_score_opt > 0:
            self.load(save_dir=save_dir)

    def marginals(self, X, batch_size=None):
        """
        Compute the marginals for the given candidates X.
        Split into batches to avoid OOM errors, then call _marginals_batch;
        defaults to no batching.
        """
        if batch_size is None:
            return self._marginals_batch(X)
        else:
            N = len(X) if self.representation else X.shape[0]
            n_batches = int(np.floor(N / batch_size))

            # Iterate over batches
            batch_marginals = []
            for b in range(0, N, batch_size):
                batch = self._marginals_batch(X[b:min(N, b+batch_size)])
                
                # Note: if a single marginal in *binary* classification is
                # returned, it will have shape () rather than (1,)- catch here
                if len(batch.shape) == 0:
                    batch = batch.reshape(1)
                    
                batch_marginals.append(batch)
            return np.concatenate(batch_marginals)

    def save(self, model_name=None, save_dir='checkpoints', verbose=True,
        global_step=0):
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
        saver.save(
            self.session,
            os.path.join(model_dir, model_name),
            global_step=global_step
        )
        if verbose:
            print("[{0}] Model saved as <{1}>".format(self.name, model_name))

    def load(self, model_name=None, save_dir='checkpoints', verbose=True):
        """Load model from file and rebuild in new graph / session."""
        model_name = model_name or self.name
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

    def _preprocess_data(self, X):
        """Generic preprocessing subclass; may be called by external methods."""
        return X
