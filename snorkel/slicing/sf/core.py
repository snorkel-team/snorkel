from typing import Any, Callable, List, Mapping, Optional

from snorkel.labeling.lf import LabelingFunction
from snorkel.preprocess import BasePreprocessor


class SlicingFunction(LabelingFunction):
    """Base class for slicing functions.

    See ``snorkel.labeling.lf.LabelingFunction`` for details.
    """

    pass


class slicing_function:
    """Decorator to define a SlicingFunction object from a function.

    Parameters
    ----------
    name
        Name of the SF
    resources
        Slicing resources passed in to ``f`` via ``kwargs``
    preprocessors
        Preprocessors to run on data points before SF execution

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
    ...     return 0 if x.a > 42 else -1
    >>> g
    SlicingFunction my_sf, Preprocessors: []
    """

    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
    ) -> None:
        if callable(name):
            raise ValueError("Looks like this decorator is missing parentheses!")
        self.name = name
        self.resources = resources
        self.pre = pre

    def __call__(self, f: Callable[..., int]) -> SlicingFunction:
        """Wrap a function to create a ``SlicingFunction``.

        Parameters
        ----------
        f
            Function that implements the core LF logic

        Returns
        -------
        SlicingFunction
            New ``SlicingFunction`` executing logic in wrapped function
        """
        name = self.name or f.__name__
        return SlicingFunction(name=name, f=f, resources=self.resources, pre=self.pre)
