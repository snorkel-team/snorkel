"""
Subpackage for Snorkel featurization.
"""

from .context_features import get_doc_count_feats, get_sentence_count_feats
from .entity_features import *
from .generic_features import feats_from_matrix_generator
from .relative_features import *
