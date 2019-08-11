from collections import Counter
from typing import Any, Callable, List, Mapping, Optional

from snorkel.preprocess import BasePreprocessor
from snorkel.preprocess.nlp import EN_CORE_WEB_SM


def check_unique_names(names: List[str]) -> None:
    """Check that operator names are unique."""
    k, ct = Counter(names).most_common(1)[0]
    if ct > 1:
        raise ValueError(f"Operator names not unique: {ct} operators with name {k}")


class base_operator_decorator:
    """Decorator to define a Snorkel operator object from a function.

    Parameters
    ----------
    name
        Name of the LF
    resources
        Labeling resources passed in to ``f`` via ``kwargs``
    preprocessors
        Preprocessors to run on data points before LF execution
    fault_tolerant
        Output ``-1`` if LF execution fails?
    """

    _operator_cls: Optional[type] = None

    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        fault_tolerant: bool = False,
    ) -> None:
        if callable(name):
            raise ValueError("Looks like this decorator is missing parentheses!")
        self.name = name
        self.resources = resources
        self.pre = pre
        self.fault_tolerant = fault_tolerant

    def __call__(self, f: Callable[..., int]) -> Callable:
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
        if self._operator_cls is None:
            raise NotImplementedError("_lf_cls must be defined")
        return self._operator_cls(
            name=name,
            f=f,
            resources=self.resources,
            pre=self.pre,
            fault_tolerant=self.fault_tolerant,
        )


class base_nlp_operator_decorator(base_operator_decorator):
    """Decorator to define a base operator child object from a function."""

    _operator_cls: Optional[type] = None

    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        fault_tolerant: bool = False,
        text_field: str = "text",
        doc_field: str = "doc",
        language: str = EN_CORE_WEB_SM,
        disable: Optional[List[str]] = None,
        memoize: bool = True,
    ) -> None:
        super().__init__(name, resources, pre, fault_tolerant)
        self.text_field = text_field
        self.doc_field = doc_field
        self.language = language
        self.disable = disable
        self.memoize = memoize

    def __call__(self, f: Callable[..., int]) -> Callable:
        """Wrap a function to create an ``BaseNLPLabelingFunction``.

        Parameters
        ----------
        f
            Function that implements the core NLP LF logic

        Returns
        -------
        BaseNLPLabelingFunction
            New ``BaseNLPLabelingFunction`` executing logic in wrapped function
        """
        if self._operator_cls is None:
            raise NotImplementedError("_lf_cls must be defined")
        name = self.name or f.__name__
        return self._operator_cls(
            name=name,
            f=f,
            resources=self.resources,
            pre=self.pre,
            fault_tolerant=self.fault_tolerant,
            text_field=self.text_field,
            doc_field=self.doc_field,
            language=self.language,
            disable=self.disable,
            memoize=self.memoize,
        )
