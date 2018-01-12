"""
Subpackage for Snorkel machine learning modules.
"""
from __future__ import absolute_import

from .utils import *
from .disc_models.rnn import reRNN, TagRNN, TextRNN
from .disc_models.logistic_regression import (
	LogisticRegression, SparseLogisticRegression
)
from .gen_learning import (
    GenerativeModel,
    GenerativeModelWeights
)