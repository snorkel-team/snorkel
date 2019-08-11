from snorkel.labeling.lf import LabelingFunction, labeling_function
from snorkel.labeling.lf.nlp import nlp_labeling_function


class SlicingFunction(LabelingFunction):
    """Base class for slicing functions.

    See ``snorkel.labeling.lf.LabelingFunction`` for details.
    """

    pass


class slicing_function(labeling_function):
    """Decorator to define a SlicingFunction object from a function.

    See ``snorkel.labeling.lf.labeling_function`` for details.
    """

    pass


class nlp_slicing_function(nlp_labeling_function):
    """Decorator to define a NLPSlicingFunction object from a function.

    See ``snorkel.labeling.lf.nlp_labeling_function`` for details.
    """

    pass
