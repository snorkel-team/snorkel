from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import os
from time import time
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from snorkel.learning.classifier import Classifier
from snorkel.learning.utils import reshape_marginals, LabelBalancer


class TorchNoiseAwareModel(Classifier, nn.Module):
    """
    Generic NoiseAwareModel class for PyTorch models.
    
    :param n_threads: Parallelism to use; single-threaded if None
    :param seed: Top level seed which is passed into both numpy operations
        via a RandomState maintained by the class, and into PyTorch
    """
    def __init__(self, n_threads=None, seed=123, **kwargs):
        Classifier.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.n_threads = n_threads
        self.seed = seed
        self.rand_state = np.random.RandomState()
        

    def _build_loss(self):
        """
        Builds the PyTorch loss for the training procedure.
        """
        # Define loss and marginals ops
        if self.cardinality > 2:
            self.loss = F.cross_entropy()
        else:
            self.loss = nn.BCEWithLogitsLoss()
    
    def _check_input(self, X):
        """Checks correctness of input; optional to implement."""
        pass
    
    def _check_model(self, lr):
        if not hasattr(self, 'loss'):
            self._build_loss()
        if not hasattr(self, 'optimizer'):
            self.optimizer = optim.Adam(self.parameters(), lr)
    
    def build_model(self, **model_kwargs):
        raise NotImplementedError
    
    def initalize_hidden_state(self):
        return None
    
    def marginals(self, X, batch_size=100):
        raise NotImplementedError

    def load(self, model_name=None, save_dir='checkpoints', verbose=True):
        """Load model from file and rebuild in new graph / session."""
        model_name = model_name or self.name
        model_dir = os.path.join(save_dir, model_name)
        warnings.warn('Unstable! Please extensively test this part of the code when time permits')
        
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

        warnings.warn('Unstable! Please extensively test this part of the code when time permits')
        # implement Path here
        print(model_dir)
        torch.save(
            self.state_dict(), 
            '{}/model.params'.format(model_dir)
        )

        if verbose:
            print("[{0}] Model saved as <{1}>".format(self.name, model_name))
        
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
        
        self.build_model(**kwargs)
        self._check_model(lr)
        
        n = len(X_train) if self.representation else X_train.shape[0]
        batch_size = min(batch_size, n)
        if verbose:
            st = time()
            print("[{0}] Training model".format(self.name))
            print("[{0}] n_train={1}  #epochs={2}  batch size={3}".format(
                self.name, n, n_epochs, batch_size
            ))

        dev_score_opt = 0.0
        
        # Run mini-batch SGD
        for epoch in range(n_epochs):
            
            epoch_losses = []
            batch_state = batch_size
            
            for batch in range(0, n, batch_size):
                
                # zero gradients for each batch
                self.optimizer.zero_grad()
                
                if batch_size < len(X_train[batch:batch+batch_size]):
                    batch_size = batch_size
                else:
                    batch_size = len(X_train[batch:batch+batch_size])

                hidden_state = self.initalize_hidden_state(batch_size)
                
                #batch Y
                batch_Y_train = torch.unsqueeze(torch.FloatTensor(Y_train[batch:batch+batch_size]), 1)
                
                # forward pass
                if hidden_state:     
                    max_batch_length = max(map(len, X_train[batch:batch+batch_size]))
                    
                    packed_X_train = torch.autograd.Variable(torch.zeros(batch_size, max_batch_length)).long()
                    for idx, seq in enumerate(X_train[batch:batch+batch_size]):
                        packed_X_train[idx, :len(seq)] = torch.LongTensor(seq)

                    output = self.forward(packed_X_train, hidden_state)
                else:
                    output = self.forward(torch.tensor(X_train))

                #Calculate loss
                calculated_loss = self.loss(output, batch_Y_train)
                
                #Compute gradient
                calculated_loss.backward()
                
                #Step on the optimizer
                self.optimizer.step()
                
                epoch_losses.append(calculated_loss)
            
            train_idxs = self.rand_state.permutation(list(range(n)))
            X_train = [X_train[j] for j in train_idxs] if self.representation \
                else X_train[train_idxs, :]
            Y_train = Y_train[train_idxs]
            
            # Print training stats and optionally checkpoint model
            if verbose and (epoch % print_freq == 0 or epoch in [0, (n_epochs-1)]):
                msg = "[{0}] Epoch {1} ({2:.2f}s)\tAverage loss={3:.6f}".format(
                    self.name, epoch, time() - st, torch.stack(epoch_losses).mean())
                if X_dev is not None:
                    scores = self.score(X_dev, Y_dev, batch_size=batch_size)
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

            batch_size = batch_state
        # Conclude training
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))
        
        # If checkpointing on, load last checkpoint (i.e. best on dev set)
        if dev_ckpt and X_dev is not None and verbose and dev_score_opt > 0:
            self.load(save_dir=save_dir)

    
        
    

    