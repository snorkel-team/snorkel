from fonduer.features.core_features import *
from fonduer.features.content_features import *
from fonduer.features.structural_features import *
from fonduer.features.table_features import *
from fonduer.features.visual_features import *


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
