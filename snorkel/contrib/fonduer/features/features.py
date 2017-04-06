from .core_features import *
from .content_features import *
from .structural_features import *
from .table_features import *
from .visual_features import *


def get_all_feats(candidate):
    for f, v in get_core_feats(candidate):
        yield f, v
    for f, v in get_content_feats(candidate):
        yield f, v
    for f, v in get_structural_feats(candidate):
        yield f, v
    for f, v in get_table_feats(candidate):
        yield f, v
    for f, v in get_visual_feats(candidate):
        yield f, v
