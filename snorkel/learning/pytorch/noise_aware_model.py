from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import os
from time import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from snorkel.learning.classifier import Classifier
from snorkel.learning.utils import reshape_marginals, LabelBalancer


def cross_entropy_loss(input, target):
    total_loss = torch.tensor(0.0)
    for i in range(input.size(1)):
        cls_idx = torch.full((input.size(0),), i, dtype=torch.long)
        loss = F.cross_entropy(input, cls_idx, reduce=False)
        total_loss += target[:, i].dot(loss)
    return total_loss / input.shape[0]


class TorchNoiseAwareModel(Classifier, nn.Module):
    """
    Generic NoiseAwareModel class for PyTorch models.
    
    :param n_threads: Parallelism to use; single-threaded if None
    """
    def __init__(self, n_threads=None, **kwargs):
        Classifier.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.n_threads = n_threads
        self.model_kwargs = None

    def _check_input(self, X):
        """Checks correctness of input; optional to implement."""
        pass
    
    def _check_model(self, lr):
        if not hasattr(self, 'loss'):
            # Define loss and marginals ops
            if self.cardinality > 2:
                self.loss = cross_entropy_loss
            else:
                self.loss = nn.BCEWithLogitsLoss()
        if not hasattr(self, 'optimizer'):
            self.optimizer = optim.Adam(self.parameters(), lr)

    def _build_model(self, **model_kwargs):
        raise NotImplementedError
    
    def marginals(self, X, batch_size=None):
        """
        Computes class probabilities for input data.

        :param X: Collection of candidates to predict labels for
        :param batch_size: Number of candidates to label at a time. Does not affect
             predicted probabilities, but useful if all candidates do not fit on GPU.
             If None, all candidates will be labeled at same time.
        :return: 1-d numpy array of probabilities that each candidate has positive label
             if class cardinality is 2. Else, a 2-d numpy array of probabilities where
             dim. 0 corresponds to candidates and dim. 1 corresponds to class labels
        """
        nn.Module.train(self, False)
        marginals = self._pytorch_outputs(X, batch_size).detach()
        return F.sigmoid(marginals).numpy() if self.cardinality == 2 else F.softmax(marginals).numpy()

    def _pytorch_outputs(self, X, batch_size):
        """
        Internal method. Computes scores, i.e., unnormalized log marginals for
        candidates.

        Public methods like marginals() and train() call _pytorch_outputs() to get scores,
        which subclasses must implement. It is a wrapper around forward() that performs
        any necessary preprocessing or batching.

        :param X: Collection of candidates to compute scores for
        :param batch_size: Number of candidates to score at a time. Does not affect
             scores, but useful if all candidates do not fit on GPU. If None, all candidates
             will be scored at same time.
        :return: 1-d tensor of scores that each candidate has positive label
             if class cardinality is 2. Else, a 2-d tensor of scores where
             dim. 0 corresponds to candidates and dim. 1 corresponds to class labels
        """
        raise NotImplementedError

    def load(self, model_name=None, save_dir='checkpoints', verbose=True):
        """Load model from file and rebuild in new graph / session."""
        model_name = model_name or self.name
        model_dir = os.path.join(save_dir, model_name)

        self.cardinality, self.name, self.model_kwargs = torch.load('{}/model.kwargs'.format(model_dir))
        self._build_model(**self.model_kwargs)
        
        self.load_state_dict(
            torch.load('{}/model.params'.format(model_dir))
        )
        if verbose:
            print("[{0}] Loaded model <{1}>".format(self.name, model_name))
        else:
            raise Exception("[{0}] No model found at <{1}>".format(
                self.name, model_name))

    def save(self, model_name=None, save_dir='checkpoints', verbose=True,
        global_step=0):
        """Save current model."""
        model_name = model_name or self.name

        # Note: Model checkpoints need to be saved in separate directories!
        model_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save((self.cardinality, self.name, self.model_kwargs), '{}/model.kwargs'.format(model_dir))
        torch.save(self.state_dict(), '{}/model.params'.format(model_dir))

        if verbose:
            print("[{0}] Model saved as <{1}>".format(self.name, model_name))
        
    def train(self, X_train, Y_train, n_epochs=25, lr=0.01, batch_size=64,
        rebalance=False, X_dev=None, Y_dev=None, print_freq=1, dev_ckpt=True,
        dev_ckpt_delay=0.75, save_dir='checkpoints', seed=123, use_cudnn=True, **kwargs):
        """
        Generic training procedure for PyTorch model

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
        :param seed: random seed used to seed PyTorch modules and random number
            generator for operations inside this method
        :param use_cudnn: Whether to use PyTorch's CuDNN backend (if available).
            Users might wish to disable it, e.g., if CuDNN training is non-deterministic
        :param kwargs: All hyperparameters that change how the network is built
            must be passed through here to be saved and reloaded to save /
            reload model. *NOTE: If a parameter needed to build the 
            network and/or is needed at test time is not included here, the
            model will not be able to be reloaded!*
        """
        # Sets random seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random_state = np.random.RandomState(seed=seed)

        # Sets whether to use CuDNN
        torch.backends.cudnn.enabled = use_cudnn

        self._check_input(X_train)
        
        verbose = print_freq > 0

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
                rand_state=random_state)
        else:
            # In categorical setting, just remove unlabeled
            diffs = Y_train.max(axis=1) - Y_train.min(axis=1)
            train_idxs = np.where(diffs > 1e-6)[0]

        self.model_kwargs = kwargs
        self._build_model(**kwargs)
        self._check_model(lr)
        
        n = len(train_idxs)
        if verbose:
            st = time()
            print("[{0}] Training model".format(self.name))
            print("[{0}] n_train={1}  #epochs={2}  batch size={3}".format(
                self.name, n, n_epochs, batch_size
            ))

        dev_score_opt = 0.0
        batch_state = batch_size

        # Run mini-batch SGD
        for epoch in range(n_epochs):
    
            # Shuffle training data
            train_idxs = random_state.permutation(list(range(n)))
            Y_train = Y_train[train_idxs]
            try:
                X_train = X_train[train_idxs, :]
            except:
                X_train = [X_train[j] for j in train_idxs]

            batch_size = min(batch_state, n) 
            epoch_losses = []

            nn.Module.train(self)
            for batch in range(0, n, batch_size):
                
                # zero gradients for each batch
                self.optimizer.zero_grad()
                
                if batch_size > len(X_train[batch:batch+batch_size]):
                    batch_size = len(X_train[batch:batch+batch_size])

                output = self._pytorch_outputs(X_train[batch:batch + batch_size], None)
                
                #Calculate loss
                calculated_loss = self.loss(output, torch.Tensor(Y_train[batch:batch+batch_size]))
                
                #Compute gradient
                calculated_loss.backward()
                
                #Step on the optimizer
                self.optimizer.step()
                
                epoch_losses.append(calculated_loss)
            
            # Print training stats and optionally checkpoint model
            if verbose and (epoch % print_freq == 0 or epoch in [0, (n_epochs-1)]):
                msg = "[{0}] Epoch {1} ({2:.2f}s)\tAverage loss={3:.6f}".format(
                    self.name, epoch+1, time() - st, torch.stack(epoch_losses).mean())
                
                if X_dev is not None:
                    scores = self.score(X_dev, Y_dev, batch_size=batch_state)
                    score = scores if self.cardinality > 2 else scores[-1]
                    score_label = "Acc." if self.cardinality > 2 else "F1"
                    msg += '\tDev {0}={1:.2f}'.format(score_label, 100. * score)
                print(msg)
                    
                # If best score on dev set so far and dev checkpointing is
                # active, save checkpoint
                if X_dev is not None and dev_ckpt and \
                    epoch > dev_ckpt_delay * n_epochs and score > dev_score_opt:
                    dev_score_opt = score
                    self.save(save_dir=save_dir, global_step=epoch)

        # Conclude training
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))
        
        # If checkpointing on, load last checkpoint (i.e. best on dev set)
        if dev_ckpt and X_dev is not None and verbose and dev_score_opt > 0:
            self.load(save_dir=save_dir)
