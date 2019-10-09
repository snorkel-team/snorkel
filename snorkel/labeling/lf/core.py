from typing import Any, Callable, List, Mapping, Optional

from snorkel.preprocess import BasePreprocessor
from snorkel.types import DataPoint


class LabelingFunction:
    """Base class for labeling functions.

    A labeling function (LF) is a function that takes a data point
    as input and produces an integer label, corresponding to a
    class. A labeling function can also abstain from voting by
    outputting ``-1``. For examples, see the Snorkel tutorials.

    This class wraps a Python function outputting a label. Extra
    functionality, such as running preprocessors and storing
    resources, is provided. Simple LFs can be defined via a
    decorator. See ``labeling_function``.

    Parameters
    ----------
    name
        Name of the LF
    f
        Function that implements the core LF logic
    resources
        Labeling resources passed in to ``f`` via ``kwargs``
    pre
        Preprocessors to run on data points before LF execution

    Raises
    ------
    ValueError
        Calling incorrectly defined preprocessors

    Attributes
    ----------
    name
        See above
    """

    def __init__(
        self,
        name: str,
        f: Callable[..., int],
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
    ) -> None:
        self.name = name
        self._f = f
        self._resources = resources or {}
        self._pre = pre or []

    def _preprocess_data_point(self, x: DataPoint) -> DataPoint:
        for preprocessor in self._pre:
            x = preprocessor(x)
            if x is None:
                raise ValueError("Preprocessor should not return None")
        return x

    def __call__(self, x: DataPoint) -> int:
        """Label data point.

        Runs all preprocessors, then passes preprocessed data point to LF.

        Parameters
        ----------
        x
            Data point to label

        Returns
        -------
        int
            Label for data point
        """
        x = self._preprocess_data_point(x)
        return self._f(x, **self._resources)

    def __repr__(self) -> str:
        preprocessor_str = f", Preprocessors: {self._pre}"
        return f"{type(self).__name__} {self.name}{preprocessor_str}"


class labeling_function:
    """Decorator to define a LabelingFunction object from a function.

    Parameters
    ----------
    name
        Name of the LF
    resources
        Labeling resources passed in to ``f`` via ``kwargs``
    preprocessors
        Preprocessors to run on data points before LF execution

    Examples
    --------
    >>> @labeling_function()
    ... def f(x):
    ...     return 0 if x.a > 42 else -1
    >>> f
    LabelingFunction f, Preprocessors: []
    >>> from types import SimpleNamespace
    >>> x = SimpleNamespace(a=90, b=12)
    >>> f(x)
    0

    >>> @labeling_function(name="my_lf")
    ... def g(x):
    ...     return 0 if x.a > 42 else -1
    >>> g
    LabelingFunction my_lf, Preprocessors: []
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

    def __call__(self, f: Callable[..., int]) -> LabelingFunction:
        """Wrap a function to create a ``LabelingFunction``.

        Parameters
        ----------
        f
            Function that implements the core LF logic

        Returns
        -------
        LabelingFunction
            New ``LabelingFunction`` executing logic in wrapped function
        """
        name = self.name or f.__name__
        return LabelingFunction(name=name, f=f, resources=self.resources, pre=self.pre)
