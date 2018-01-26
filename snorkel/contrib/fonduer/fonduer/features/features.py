from __future__ import unicode_literals
from .core_features import *
from .content_features import *
from .structural_features import *
from .table_features import *
from .visual_features import *


def get_all_feats(candidates):
    for id, f, v in get_core_feats(candidates):
        yield id, f, v
    for id, f, v in get_content_feats(candidates):
        yield id, f, v
    for id, f, v in get_structural_feats(candidates):
        yield id, f, v
    for id, f, v in get_table_feats(candidates):
        yield id, f, v
    for id, f, v in get_visual_feats(candidates):
        yield id, f, v
