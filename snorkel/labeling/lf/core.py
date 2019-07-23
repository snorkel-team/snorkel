from typing import Any, Callable, List, Mapping, Optional

from snorkel.labeling.preprocess import BasePreprocessor
from snorkel.types import DataPoint, Label


class LabelingFunction:
    """Base class for labeling functions.

    A labeling function (LF) is a function that takes a data point
    as input and produces an integer label, corresponding to a
    class. A labeling function can also abstain from voting by
    outputting ``None``. For examples, see the Snorkel tutorials.

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
    preprocessors
        Preprocessors to run on data points before LF execution
    fault_tolerant
        Output ``None`` if LF execution fails?

    Raises
    ------
    ValueError
        Calling incorrectly defined preprocessors

    Attributes
    ----------
    name
        See above
    fault_tolerant
        See above
    """

    def __init__(
        self,
        name: str,
        f: Callable[..., Label],
        resources: Optional[Mapping[str, Any]] = None,
        preprocessors: Optional[List[BasePreprocessor]] = None,
        fault_tolerant: bool = False,
    ) -> None:
        self.name = name
        self.fault_tolerant = fault_tolerant
        self._f = f
        self._resources = resources or {}
        self._preprocessors = preprocessors or []

    def _preprocess_data_point(self, x: DataPoint) -> DataPoint:
        for preprocessor in self._preprocessors:
            x = preprocessor(x)
            if x is None:
                raise ValueError("Preprocessor should not return None")
        return x

    def __call__(self, x: DataPoint) -> Label:
        """Label data point.

        Runs all preprocessors, then passes to LF. If an exception
        is encountered and the LF is in fault tolerant mode,
        the LF abstains from voting.

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
        if self.fault_tolerant:
            try:
                return self._f(x, **self._resources)
            except Exception:
                return None
        return self._f(x, **self._resources)

    def __repr__(self) -> str:
        preprocessor_str = f", Preprocessors: {self._preprocessors}"
        return f"{type(self).__name__} {self.name}{preprocessor_str}"


class labeling_function:
    """Decorator to define a LabelingFunction object from a function.

    Parameters
    ----------
    See ``LabelingFunction``.

    Examples
    --------
    >>> @labeling_function()
    ... def f(x):
    ...     if x.a > 42: return 1
    >>> f
    LabelingFunction f, Preprocessors: []

    >>> @labeling_function(name="my_lf")
    ... def g(x):
    ...     if x.a > 42: return 1
    >>> g
    LabelingFunction my_lf, Preprocessors: []
    """

    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        preprocessors: Optional[List[BasePreprocessor]] = None,
        fault_tolerant: bool = False,
    ) -> None:
        self.name = name
        self.resources = resources
        self.preprocessors = preprocessors
        self.fault_tolerant = fault_tolerant

    def __call__(self, f: Callable[..., Label]) -> LabelingFunction:
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
        return LabelingFunction(
            name=name,
            f=f,
            resources=self.resources,
            preprocessors=self.preprocessors,
            fault_tolerant=self.fault_tolerant,
        )
