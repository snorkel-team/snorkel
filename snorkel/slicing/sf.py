"""Define SF appliers as simple alias of LF counterparts for now"""
from snorkel.labeling.lf import LabelingFunction, labeling_function


class SlicingFunction(LabelingFunction):
    pass


class slicing_function(labeling_function):
    pass
