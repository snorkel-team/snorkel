from snorkel.labeling.lf import LabelingFunction
from snorkel.labeling.lf.nlp import NLPLabelingFunction
from snorkel.utils.data_operators import (
    base_nlp_operator_decorator,
    base_operator_decorator,
)


class SlicingFunction(LabelingFunction):
    """Base class for slicing functions.

    See ``snorkel.labeling.lf.LabelingFunction`` for details.
    """

    pass


class slicing_function(base_operator_decorator):
    """Decorator to define a SlicingFunction object from a function.

    Parameters
    ----------
    name
        Name of the SF
    resources
        Slicing resources passed in to ``f`` via ``kwargs``
    preprocessors
        Preprocessors to run on data points before SF execution
    fault_tolerant
        Output ``-1`` if LF execution fails?

    Examples
    --------
    >>> @slicing_function()
    ... def f(x):
    ...     return x.a > 42
    >>> f
    SlicingFunction f, Preprocessors: []
    >>> from types import SimpleNamespace
    >>> x = SimpleNamespace(a=90, b=12)
    >>> f(x)
    True

    >>> @slicing_function(name="my_sf")
    ... def g(x):
    ...     return x.a > 42
    >>> g
    SlicingFunction my_sf, Preprocessors: []
    """

    _operator_cls = SlicingFunction


class NLPSlicingFunction(NLPLabelingFunction):
    """Base class for slicing functions.

    See ``snorkel.labeling.lf.LabelingFunction`` for details.
    """

    pass


class nlp_slicing_function(base_nlp_operator_decorator):
    """Decorator to define a NLPSlicingFunction object from a function.

    See ``snorkel.labeling.lf.nlp_labeling_function`` for details.
    """

    _operator_cls = NLPSlicingFunction
