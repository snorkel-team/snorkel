import numpy as np

from six.moves import range

from .utils import MentionScorer
from ..annotations import save_marginals


class Classifier(object):
    """Simple abstract base class for a probabilistic classifier."""

    # Set this class variable to True if train, marginals, predict, and score,
    # take a list of @Candidates as the first argument X;
    # otherwise assume X is an AnnotationMatrix
    representation = False

    def __init__(self, cardinality=2, name=None, seed=None):
        self.name = name or self.__class__.__name__
        self.cardinality = cardinality
        self.seed = seed

    def marginals(self, X, **kwargs):
        raise NotImplementedError()

    def save_marginals(self, session, X, training=False):
        """Save the predicted marginal probabilities for the Candidates X."""
        save_marginals(session, X, self.marginals(X), training=training)

    def predictions(self, X, b=0.5):
        """Return numpy array of elements in {-1,0,1}
        based on predicted marginal probabilities.
        """
        if self.cardinality > 2:
            return self.marginals(X).argmax(axis=1) + 1
        else:
            return np.array([1 if p > b else -1 if p < b else 0 
                for p in self.marginals(X)])

    def score(self, X_test, Y_test, b=0.5, set_unlabeled_as_neg=True):
        """
        Returns the summary scores:
            * For binary: precision, recall, F1 score
            * For categorical: accuracy
        
        :param X_test: The input test candidates, as a list or annotation matrix
        :param Y_test: The input test labels, as a list or annotation matrix
        :param b: Decision boundary *for binary setting only*
        :param set_unlabeled_as_neg: Whether to map 0 labels -> -1, *binary setting.*

        Note: Unlike in self.error_analysis, this method assumes X_test and
        Y_test are properly collated!
        """
        predictions = self.predictions(X_test, b=b)

        # Convert Y_test to dense numpy array
        try:
            Y_test = np.array(Y_test.todense()).reshape(-1)
        except:
            Y_test = np.array(Y_test)

        # Compute accuracy for categorical, or P/R/F1 for binary settings
        if self.cardinality > 2:
            # Compute and return accuracy
            correct = np.where([predictions == Y_test])[0].shape[0]
            return correct / float(Y_test.shape[0])
        else:
            # Either remap or filter out unlabeled (0-valued) test labels
            if set_unlabeled_as_neg:
                Y_test[Y_test == 0] = -1
            else:
                predictions = predictions[Y_test != 0]
                Y_test = Y_test[Y_test != 0]
            
            # Compute and return precision, recall, and F1 score
            tp = (0.5 * (predictions * Y_test + 1))[predictions == 1].sum()
            pred_pos = predictions[predictions == 1].sum()
            prec = tp / float(pred_pos) if pred_pos > 0 else 0.0
            pos = Y_test[Y_test == 1].sum()
            rec = tp / float(pos) if pos > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
            return prec, rec, f1

    def error_analysis(self, session, X_test, Y_test, 
        gold_candidate_set=None, b=0.5, set_unlabeled_as_neg=True, display=True,
        scorer=MentionScorer, **kwargs):
        """
        Prints full score analysis using the Scorer class, and then returns the
        a tuple of sets conatining the test candidates bucketed for error 
        analysis, i.e.:
            * For binary: TP, FP, TN, FN
            * For categorical: correct, incorrect
        
        :param X_test: The input test candidates, as a list or annotation matrix
        :param Y_test: The input test labels, as a list or annotation matrix
        :param gold_candidate_set: Full set of TPs in the test set
        :param b: Decision boundary *for binary setting only*
        :param set_unlabeled_as_neg: Whether to map 0 labels -> -1, *binary setting*
        :param display: Print score report
        :param scorer: The Scorer sub-class to use
        """
        # Compute the marginals
        test_marginals = self.marginals(X_test, **kwargs)

        # Get the test candidates
        test_candidates = [
            X_test.get_candidate(session, i) for i in range(X_test.shape[0])
        ] if not self.representation else X_test

        # Initialize and return scorer
        s = scorer(test_candidates, Y_test, gold_candidate_set)          
        return s.score(test_marginals, train_marginals=None, b=b,
            display=display, set_unlabeled_as_neg=set_unlabeled_as_neg)

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()
