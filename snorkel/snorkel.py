import warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import scipy.sparse as sparse
from learning import LogReg, odds_to_prob
from lstm import *
from learning_utils import test_scores, calibration_plots, training_set_summary_stats, sparse_abs, LF_coverage, \
    LF_overlaps, LF_conflicts, LF_accuracies
from pandas import Series, DataFrame


class TrainingSet(object):
    """
    Wrapper data object which applies the LFs to the candidates comprising the training set,
    featurizes them, and then stores the resulting _noisy training set_, as well as the LFs and featurizer.

    As input takes:
        - A set of Candidate objects comprising the training set
        - A set of labeling functions (LFs) which are functions f : Candidate -> {-1,0,1}
        - A Featurizer object, which is applied to the Candidate objects to generate features
    """
    def __init__(self, training_candidates, lfs, featurizer=None):
        self.training_candidates = training_candidates
        self.featurizer          = featurizer
        self.lfs                 = lfs
        self.lf_names            = [lf.__name__ for lf in lfs]
        self.L, self.F           = self.transform(self.training_candidates, fit=True)

        # Cached data for LF empirical stats
        self.lf_stat_candidates = None
        self.lf_stat_labels     = None
        self.Ls                 = None

        # Print training set summary stats
        self.summary_stats()

    def transform(self, candidates, fit=False):
        """Apply LFs and featurize the candidates"""
        print "Applying LFs..."
        L = self._apply_lfs(candidates)
        F = None
        if self.featurizer is not None:
            print "Featurizing..."
            F = self.featurizer.fit_transform(candidates) if fit else self.featurizer.transform(candidates)
        return L, F

    def _apply_lfs(self, candidates):
        """Apply the labeling functions to the candidates to populate X"""
        X = sparse.lil_matrix((len(candidates), len(self.lfs)))
        for i,c in enumerate(candidates):
            for j,lf in enumerate(self.lfs):
                X[i,j] = lf(c)
        return X.tocsr()

    def summary_stats(self, return_vals=False, verbose=True):
        """Print out basic stats about the LFs wrt the training candidates"""
        return training_set_summary_stats(self.L, return_vals=return_vals, verbose=verbose)

    def lf_stats(self, candidates=None, labels=None):
        """Returns a pandas Dataframe with the LFs and various per-LF statistics"""
        N, M = self.L.shape

        # Default LF stats
        d = {
            'j'         : range(len(self.lfs)),
            'coverage'  : Series(data=LF_coverage(self.L), index=self.lf_names),
            'overlaps'  : Series(data=LF_overlaps(self.L), index=self.lf_names),
            'conflicts' : Series(data=LF_conflicts(self.L), index=self.lf_names)
        }

        # Empirical stats, based on supplied development set
        if candidates and labels is not None:
            if self.Ls is None or candidates != self.lf_stat_candidates or any(labels != self.lf_stat_labels):
                self.lf_stat_candidates = candidates
                self.lf_stat_labels     = labels
                self.Ls          = self._apply_lfs(candidates)
            d['accuracy'] = Series(data=LF_accuracies(self.Ls, self.lf_stat_labels), index=self.lf_names)
        return DataFrame(data=d, index=self.lf_names)


class Learner(object):
    """
    Core learning class for Snorkel, encapsulating the overall process of learning a generative model of the
    training data set (specifically: of the LF-emitted labels and the true class labels), and then using this
    to train a given noise-aware discriminative model.

    As input takes a TrainingSet object and a NoiseAwareModel object (the discriminative model to train).
    """
    def __init__(self, training_set, model=None):
        self.training_set = training_set
        self.model        = model

        # We need to know certain properties _that are set in the model defn_
        self.bias_term = self.model.bias_term if hasattr(self.model, 'bias_term') else False

        # Derived objects from the training set
        self.L_train         = self.training_set.L
        self.X_train         = None
        self.n_train, self.m = self.L_train.shape
        if self.training_set.F is not None:
            self.F_train = self.training_set.F
            self.f       = self.F_train.shape[1]

        # Cache the transformed test set as well
        self.test_candidates = None
        self.gold_labels     = None
        self.L_test          = None
        self.F_test          = None
        self.X_test          = None

    def _set_model_X(self, L, F):
        """Given LF matrix L, feature matrix F, return the matrix used by the end discriminative model."""
        n, m = L.shape
        X    = sparse.hstack([L, F], format='csr')
        if self.bias_term:
            X = sparse.hstack([X, np.ones((n, 1))], format='csr')
        return X

    def train(self, lf_w0=5.0, feat_w0=0.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        # Set the initial weights for LFs and feats
        w0 = np.concatenate([lf_w0*np.ones(self.m), feat_w0*np.ones(self.f)])
        w0 = np.append(w0, 0) if self.bias_term else w0

        # Construct matrix X for "joint" approach
        self.X_train = self._set_model_X(self.L_train, self.F_train)

        # Train model
        self.model.train(self.X_train, w0=w0, **model_hyperparams)

    def test(self, test_candidates, gold_labels, display=True, return_vals=False):
        """
        Apply the LFs and featurize the test candidates, using the same transformation as in training set;
        then test against gold labels using trained model.
        """
        gold_labels = np.array(gold_labels)

        # Cache transformed test set
        if self.X_test is None or test_candidates != self.test_candidates or any(gold_labels != self.gold_labels):
            self.test_candidates     = test_candidates
            self.gold_labels         = gold_labels
            self.L_test, self.F_test = self.training_set.transform(test_candidates)
            self.X_test              = self._set_model_X(self.L_test, self.F_test)
        if display:
            calibration_plots(self.model.marginals(self.X_train), self.model.marginals(self.X_test), gold_labels)
        return test_scores(self.model.predict(self.X_test), gold_labels, return_vals=return_vals, verbose=display)

    def lf_weights(self):
        return self.model.w[:self.m]

    def lf_accs(self):
        return odds_to_prob(self.lf_weights())

    def feature_weights(self):
        return self.model.w[self.m:self.m+self.f]

    def predictions(self):
        return self.model.predict(self.X_test)

    def test_mv(self, test_candidates, gold_labels, display=True, return_vals=False):
        """Test *unweighted* majority vote of *just the LFs*"""
        # Ensure that L_test is initialized
        self.test(test_candidates, gold_labels, display=False)

        # L_test * 1
        mv_pred = np.ravel(np.sign(self.L_test.sum(axis=1)))
        return test_scores(mv_pred, gold_labels, return_vals=return_vals, verbose=display)

    def test_wmv(self, test_candidates, gold_labels, display=True, return_vals=False):
        """Test *weighted* majority vote of *just the LFs*"""
        # Ensure that L_test is initialized
        self.test(test_candidates, gold_labels, display=False)

        # L_test * w_lfs
        wmv_pred = np.sign(self.L_test.dot(self.lf_weights()))
        return test_scores(wmv_pred, gold_labels, return_vals=return_vals, verbose=display)

    def feature_stats(self, n_max=100):
        """Return a DataFrame of highest (abs)-weighted features"""
        idxs = np.argsort(np.abs(self.feature_weights()))[::-1][:n_max]
        d = {'j': idxs, 'w': [self.feature_weights()[i] for i in idxs]}
        return DataFrame(data=d, index=[self.training_set.featurizer.inv_index[i] for i in idxs])


class PipelinedLearner(Learner):
    """
    Implements the **"pipelined" approach**- this is the method more literally corresponding
    to the Data Programming paper
    """
    def _set_model_X(self, L, F):
        n, f = F.shape
        X    = F.tocsr()
        if self.bias_term:
            X = sparse.hstack([X, np.ones((n, 1))], format='csr')
        return X

    def train_lf_model(self, w0=1.0, **model_hyperparams):
        """Train the first _generative_ model of the LFs"""
        w0 = w0*np.ones(self.m)
        self.training_model = LogReg()
        self.training_model.train(self.L_train, w0=w0, **model_hyperparams)

        # Compute marginal probabilities over the candidates from this model of the training set
        return self.training_model.marginals(self.L_train)

    def train_model(self, training_marginals, w0=0.0, **model_hyperparams):
        """Train the provided end _discriminative_ model"""
        w0           = w0*np.ones(self.f)
        w0           = np.append(w0, 0) if self.bias_term else w0
        self.X_train = self._set_model_X(self.L_train, self.F_train)
        self.w       = self.model.train(self.X_train, training_marginals=training_marginals, \
                        w0=w0, **model_hyperparams)

    def train(self, feat_w0=0.0, lf_w0=1.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        print "Training LF model..."
        training_marginals = self.train_lf_model(w0=lf_w0, **model_hyperparams)
        print "Training model..."
        self.train_model(training_marginals, w0=feat_w0, **model_hyperparams)

    def lf_weights(self):
        return self.training_model.w

    def feature_weights(self):
        return self.model.w


class RepresentationLearner(PipelinedLearner):
    """
    Implements the _pipelined_ approach for an end model that also learns a representation
    """
    def train_model(self, training_marginals, w0=0.0, **model_hyperparams):
        """Train the provided end _discriminative_ model"""
        self.w = self.model.train(self.training_set.training_candidates, training_marginals=training_marginals, \
                                  **model_hyperparams)

    def test(self, test_candidates, gold_labels, display=True, return_vals=False):
        # Cache transformed test set
        if test_candidates != self.test_candidates or any(gold_labels != self.gold_labels):
            self.test_candidates = test_candidates
            self.gold_labels     = gold_labels
        if display:
            calibration_plots(self.model.marginals(self.training_set.training_candidates), \
                                self.model.marginals(self.test_candidates), gold_labels)
        return test_scores(self.model.predict(self.test_candidates), gold_labels, return_vals=return_vals, verbose=display)

    def predictions(self):
        return self.model.predict(self.test_candidates)
