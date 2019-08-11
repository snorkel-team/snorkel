from typing import Any, Callable, List, Mapping, Optional

from snorkel.labeling.lf import labeling_function
from snorkel.labeling.lf.nlp import NLPLabelingFunction
from snorkel.preprocess import BasePreprocessor
from snorkel.preprocess.nlp import EN_CORE_WEB_SM


class NLPSlicingFunction(NLPLabelingFunction):
    """Base class for slicing functions.

    See ``snorkel.labeling.lf.nlp.NLPLabelingFunction`` for details.
    """

    pass


class nlp_slicing_function(labeling_function):
    """Decorator to define a NLPSlicingFunction child object from a function."""

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

    def __call__(self, f: Callable[..., int]) -> NLPSlicingFunction:
        """Wrap a function to create an ``NLPSlicingFunction``.

        Parameters
        ----------
        f
            Function that implements the core NLP LF logic

        Returns
        -------
        NLPSlicingFunction
            New ``NLPSlicingFunction`` executing logic in wrapped function
        """
        name = self.name or f.__name__
        return NLPSlicingFunction(
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
