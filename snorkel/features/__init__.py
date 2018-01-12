"""
Submodule for Snorkel featurization.
"""
from __future__ import absolute_import

from .context_features import (
	get_document_token_count_feats,
	get_sentence_token_count_feats,
)
from .entity_features import *
from .generic_features import get_feats_from_matrix
from .relative_features import (
	get_document_relative_frequency_feats,
	get_first_document_span_feats,
	get_first_document_span_feats_stopwords,
	get_sentence_relative_frequency_feats,
	get_span_feats,
	get_span_feats_stopwords,
	get_span_splits,
	get_span_splits_stopwords,
)

