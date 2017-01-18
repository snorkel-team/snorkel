from __future__ import print_function

import concurrent.futures
import numba
import numpy as np
import time

from collections import defaultdict
from disc_learning import NoiseAwareModel
from math import exp, log


MIN_LR = 1e-6


class FMCT(NoiseAwareModel):
    """fastmulticontext"""
    def __init__(self, preprocess_function=None):
        self.fmct         = None
        self.w            = None
        self.X_train      = None
        self.preprocess_f = preprocess_function

    def train(self, training_marginals, embed_matrices, **hyperparams):
        """
        Train method for fastmulticontext
        training_marginals: marginal probabilities for training examples
        embed_matrices: list of matrices to embed
        hyperparams: fmct hyperparams, including raw_xs
        """
        self.fmct = fastmulticontext(self.preprocess_f)
        self.fmct.train(training_marginals, embed_matrices, **hyperparams)

    def marginals(self, embed_matrices, raw_xs=None):
        return self.fmct.predict(embed_matrices, raw_xs)


def get_matrix_keys(matrices):
    n, m = matrices[0].shape[0], len(matrices)
    embed_xs = [[[] for _ in xrange(m)] for _ in xrange(n)]
    for k, matrix in enumerate(matrices):
        matrix_coo = matrix.tocoo(copy=True)
        for i, j, ct in zip(matrix_coo.row, matrix_coo.col, matrix_coo.data):
            embed_xs[i][k].extend('FEATURE_{0}_{1}'.format(k, j) for _ in xrange(int(ct)))
        print("Processed {0} matrices".format(k))
    return embed_xs


@numba.jit(nopython=True, nogil=True)
def fmct_activation(z, hidden_embed, wo, wo_raw, wi_sub, x_ct, x_type, x_raw):
    """
    JIT function for computing activation for fmct
    """
    n_classes, embed_size = wo.shape
    raw_size = wo_raw.shape[1]
    x_size, dim = wi_sub.shape
    # Embedded features
    for i in xrange(x_size):
        if x_ct[i] == 0:
            continue
        for j in xrange(dim):
            hidden_embed[j + x_type[i]*dim] += wi_sub[i][j] / x_ct[i]
    # Compute activations
    for k in xrange(n_classes):
        for j in xrange(embed_size):
            z[k] += wo[k][j] * hidden_embed[j]
        for r in xrange(raw_size):
            z[k] += wo_raw[k][r] * x_raw[r]


@numba.jit(nopython=True, nogil=True)
def fmct_update(wo, wo_raw, wi_sub, x_ct, x_type, x_raw, p, lr, lambda_n):
    """
    JIT function for issuing SGD step of fmct
    """
    n_classes, embed_size = wo.shape
    raw_size = wo_raw.shape[1]
    x_size, dim = wi_sub.shape
    # Get activations
    z = np.zeros(n_classes)
    hidden_embed = np.zeros(embed_size)
    fmct_activation(z, hidden_embed, wo, wo_raw, wi_sub, x_ct, x_type, x_raw)
    # Compute softmax
    mz = z[0]
    for i in xrange(n_classes):
        mz = max(mz, z[i])
    s = 0
    for k in xrange(n_classes):
        z[k] = exp(z[k] - mz)
        s += z[k]
    for k in xrange(n_classes):
        z[k] /= s
    # Update embedding gradient and linear layer
    grad = np.zeros(embed_size)
    for k in xrange(n_classes):
        # Noise-aware gradient calculation
        # g(x) = [(1-p)\hat{p} - p(1-\hat{p})]x
        alpha = lr * ((1.0-p[k])*z[k] - p[k]*(1.0-z[k]))
        # Updates for embedded features
        for j in xrange(embed_size):
            grad[j] += alpha * wo[k][j]
            # Apply regularization first
            wo[k][j] *= (1.0 - lr * lambda_n)
            wo[k][j] -= alpha * hidden_embed[j]
        # Updates for raw features
        for r in xrange(raw_size):
            # Apply regularization first
            wo_raw[k][r] *= (1.0 - lr * lambda_n)
            wo_raw[k][r] -= alpha * x_raw[r]
    # Update embeddings
    for i in xrange(x_size):
        for j in xrange(dim):
            if x_ct[i] == 0:
                continue
            # Apply regularization first
            wi_sub[i][j] *= (1.0 - lr * lambda_n)
            wi_sub[i][j] -= (grad[j + x_type[i]*dim] / x_ct[i])
    # Return loss
    pmx, lmx = 0.0, None
    for k in xrange(n_classes):
        if p[k] > pmx:
            pmx, lmx = p[k], -log(z[k])
    return lmx


@numba.jit(nopython=True, nogil=True)
def print_status(progress, loss, n_examples, lr):
    """ Print training progress and loss """
    print('-------------------')
    print(100. * progress)
    print(loss / n_examples)
    print('-------------------')


@numba.jit(nopython=True, nogil=True)
def fmct_sgd_thread(thread_n, wo, wo_raw, wi, marginals, lambda_n, epoch, n, lr, raw_xs,
                    n_print, feat_start, feat_end, f_cache, f_ct_cache, f_t_cache):
    
    loss, n_examples, lr_orig = 0, 0, lr

    ### Run SGD ###
    for kt in xrange(epoch * n):
        # Update status and learning rate
        k = kt % n
        n_examples += 1
        progress = float(kt) / (epoch * n)
        lr = max(MIN_LR, lr_orig * (1.0 - progress))
        # Retrieve features and probabilities
        feats = f_cache[feat_start[k] : feat_end[k]]
        feats_ct = f_ct_cache[feat_start[k] : feat_end[k]]
        feats_type = f_t_cache[feat_start[k] : feat_end[k]]
        raw_feats = raw_xs[k]
        if len(feats) + len(raw_feats) == 0:
            continue
        # Gradient step
        wi_sub = wi[feats]
        loss += fmct_update(
            wo, wo_raw, wi_sub, feats_ct, feats_type, raw_feats,
            marginals[k], lr, lambda_n,
        )
        wi[feats, :] = wi_sub
        # Update learning rate and print status
        if thread_n == 0 and kt % n_print == 0:
            print_status(progress, loss, n_examples, lr)
            
    if thread_n == 0:
        print_status(1, loss, n_examples, lr)
        print('\n')


def fmct_sgd(n_threads, *args):
    if n_threads == 1:
        fmct_sgd_thread(0, *args)
    else:
        threadpool = concurrent.futures.ThreadPoolExecutor(n_threads)
        threads = [
            threadpool.submit(fmct_sgd_thread, i, *args) for i in xrange(n_threads)
        ]
        concurrent.futures.wait(threads)
        for thread in threads:
            if thread.exception() is not None:
                raise thread.exception()

class fastmulticontext(object):
    
    def __init__(self, preprocess_function=get_matrix_keys):
        """
        Initialize fastmulticontext model
        preprocess_function: function returning features for embedding sequence
        """
        self.vocabs       = []
        self.n_classes    = None
        self.n_embed      = None
        self.vocab_slice  = None
        self.wo           = None
        self.wo_raw       = None
        self.wi           = None
        self.preprocess_f = preprocess_function
        
    def train(self, marginals, embed_xs, raw_xs=None, dim=50, lr=0.05, lambda_l2=1e-7,
              epoch=5, min_ct=1, n_print=10000, n_threads=16, seed=1701):
        """
        Train FMCT model
        marginals: marginal probabilities for training examples (array)
        embed_xs: embedded features for training examples (passed to feat_f)
        raw_xs: raw features for training examples (2d numpy array)
        dim: dimensionality of embeddings
        lr: initial learning rate
        epoch: number of learning epochs
        min_ct: minimum feature count for modeling
        n_print: how frequently to print updates
        """

        if seed is not None:
            np.random.seed(seed=seed)

        print("Processing data", end='\t\t')
        embed_xs = self.preprocess_f(embed_xs) if self.preprocess_f else embed_xs
        self.n_classes = 2 # Hardcode binary classification for now
        n = len(embed_xs)

        ### Init feature indices ###
        self.n_embed = len(embed_xs[0])
        # If no raw features, add a bias term
        if raw_xs is None:
            raw_xs = np.ones((n, 1))

        ### Build vocab ###
        print("Building vocab", end='\t\t')
        self._build_vocabs(embed_xs, min_ct)
        all_vocab_size = self.vocab_slice[-1]
        feat_cache = []
        feat_ct_cache = []
        feat_type_cache = []
        feat_start, feat_end = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        s = 0
        for k in xrange(n):
            feats, feats_ct, feats_type = self._get_vocab_index(embed_xs[k])
            feat_cache.extend(feats)
            feat_ct_cache.extend(feats_ct)
            feat_type_cache.extend(feats_type)
            feat_start[k] = s
            feat_end[k] = s + len(feats)
            s += len(feats)
        feat_cache = np.ravel(feat_cache).astype(int)
        feat_ct_cache = np.ravel(feat_ct_cache)
        feat_type_cache = np.ravel(feat_type_cache).astype(int)
        
        ### Init model ###
        print("Training")
        self.wo = np.zeros((self.n_classes, dim * self.n_embed))
        self.wo_raw = np.zeros((self.n_classes, raw_xs.shape[1]))
        self.wi = np.random.uniform(-1.0 / dim, 1.0 / dim, (all_vocab_size, dim))
        marginals = np.array([[1.0 - float(p), float(p)] for p in marginals])
        lambda_n = float(lambda_l2) / n

        s = time.time()
        fmct_sgd(
            n_threads, self.wo, self.wo_raw, self.wi, marginals, lambda_n,
            epoch, n, lr, raw_xs, n_print, feat_start, feat_end, feat_cache,
            feat_ct_cache, feat_type_cache,
        )
        print("Training time: {0:.3f} seconds".format(time.time() - s))

    def predict(self, embed_xs, raw_xs=None):
        """
        Predict marginals for new examples
        embed_xs: embedded features
        raw_xs: raw features
        """
        embed_xs = self.preprocess_f(embed_xs) if self.preprocess_f else embed_xs
        n = len(embed_xs)
        log_odds = np.zeros(n)
        n_skipped = 0
        # If no raw features, add a bias term
        if raw_xs is None:
            raw_xs = np.ones((n, 1))
        for k in xrange(n):
            x, x_raw = embed_xs[k], raw_xs[k, :]
            feats, feats_ct, feats_type = self._get_vocab_index(x)
            if len(feats) + np.sum(x_raw) == 0:
                n_skipped += 1
                log_odds[k] = 0.0
                continue
            wi_sub = self.wi[feats, :]
            z = np.zeros(self.n_classes)
            hidden_embed = np.zeros(self.wo.shape[1])
            fmct_activation(
                z, hidden_embed, self.wo, self.wo_raw, wi_sub, feats_ct, feats_type, x_raw
            )
            log_odds[k] = z[1]
        print("Skipped {0} because no feats".format(n_skipped))
        return 1.0 / (1.0 + np.exp(-log_odds))        
    
    def _build_vocabs(self, embed_xs, min_ct):
        """
        Build vocabulary
        embed_xs: features to embed
        min_ct: minimum count of feature to include in modeling
        """
        if not hasattr(min_ct, '__iter__'):
            min_ct = [min_ct for _ in xrange(self.n_embed)]
        count_dicts = [defaultdict(int) for _ in xrange(self.n_embed)]
        # Count instances of feats in corpus
        for x in embed_xs:
            for d, feats in enumerate(x):
                for feat in feats:
                    count_dicts[d][feat] += 1
        # Build vocab from feats with sufficient counts
        self.vocabs = [{} for _ in xrange(self.n_embed)]
        for d, count_dict in enumerate(count_dicts):
            for feat, ct in count_dict.iteritems():
                if ct >= min_ct[d]:
                    self.vocabs[d][feat] = len(self.vocabs[d])
            print("Built vocab {0} of length {1}".format(d, len(self.vocabs[d])))
        self.vocab_slice = [0]
        for vocab in self.vocabs:
            self.vocab_slice.append(self.vocab_slice[-1] + len(vocab))
    
    def _get_vocab_index(self, x):
        """
        Retrieve feat indices of x
        x: feature to embed
        """
        # Get feature indices in each vocab
        vocab_idxs = []
        for d, feats in enumerate(x):
            indices = []
            for feat in feats:
                if feat in self.vocabs[d]:
                    indices.append(self.vocabs[d][feat])
            vocab_idxs.append(np.ravel(sorted(indices)))
        # Aggregate to global index
        m, s = np.sum([len(vc) for vc in vocab_idxs]), 0
        feat_idxs, feat_cts, feat_type = np.zeros(m), np.zeros(m), np.zeros(m)
        for i, vc in enumerate(vocab_idxs):
            feat_idxs[s : s+len(vc)] = (vc + self.vocab_slice[i])
            feat_cts[s : s+len(vc)] = len(vc)
            feat_type[s : s+len(vc)] = i
            s += len(vc)
        return feat_idxs.astype(int), feat_cts.astype(int), feat_type.astype(int)
    
    def _print_status(self, progress, loss, n_examples, lr):
        """ Print training progress and loss """
        sys.stdout.write("\rProgress: {0:06.3f}%\tLoss: {1:.6f}\tLR={2:.6f}".format(
            100. * progress, loss / n_examples, lr
        ))
        sys.stdout.flush()