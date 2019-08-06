from typing import Any, Callable, List, Mapping, NamedTuple, Optional

from snorkel.labeling.preprocess import BasePreprocessor
from snorkel.labeling.preprocess.nlp import EN_CORE_WEB_SM, SpacyPreprocessor

from .core import LabelingFunction, labeling_function


class SpacyPreprocessorParameters(NamedTuple):
    """Parameters needed to construct a SpacyPreprocessor.

    See ``snorkel.labeling.preprocess.nlp.SpacyPreprocessor``.
    """

    text_field: str
    doc_field: str
    language: str
    disable: Optional[List[str]]
    preprocessors: List[BasePreprocessor]
    memoize: bool


class SpacyPreprocessorConfig(NamedTuple):
    """Tuple of SpacyPreprocessor and the parameters used to construct it."""

    nlp: SpacyPreprocessor
    parameters: SpacyPreprocessorParameters


class NLPLabelingFunction(LabelingFunction):
    """Special labeling function type for SpaCy-based LFs.

    This class is a special version of ``LabelingFunction``. It
    has a ``SpacyPreprocessor`` integrated which shares a cache
    with all other ``NLPLabelingFunction`` instances. This makes
    it easy to define LFs that have a text input field and have
    logic written over SpaCy ``Doc`` objects. Examples passed
    into an ``NLPLabelingFunction`` will have a new field which
    can be accessed which contains a SpaCy ``Doc``. By default,
    this field is called ``doc``. A ``Doc`` object is
    a sequence of ``Token`` objects, which contain information
    on lemmatization, parts-of-speech, etc. ``Doc`` objects also
    contain fields like ``Doc.ents``, a list of named entities,
    and ``Doc.noun_chunks``, a list of noun phrases. For details
    of SpaCy ``Doc`` objects and a full attribute listing,
    see https://spacy.io/api/doc.

    Simple ``NLPLabelingFunction``\s can be defined via a
    decorator. See ``nlp_labeling_function``.

    Parameters
    ----------
    name
        Name of the LF
    f
        Function that implements the core LF logic
    resources
        Labeling resources passed in to ``f`` via ``kwargs``
    preprocessors
        Preprocessors to run before SpacyPreprocessor is executed
    fault_tolerant
        Output -1 if LF execution fails?
    text_field
        Name of data point text field to input
    doc_field
        Name of data point field to output parsed document to
    language
        SpaCy model to load
        See https://spacy.io/usage/models#usage
    disable
        List of pipeline components to disable
        See https://spacy.io/usage/processing-pipelines#disabling
    memoize
        Memoize preprocessor outputs?

    Raises
    ------
    ValueError
        Calling incorrectly defined preprocessors

    Example
    -------
    >>> def f(x):
    ...     person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
    ...     return 0 if len(person_ents) > 0 else -1
    >>> has_person_mention = NLPLabelingFunction(name="has_person_mention", f=f)
    >>> has_person_mention
    NLPLabelingFunction has_person_mention, Preprocessors: [SpacyPreprocessor...]

    >>> from types import SimpleNamespace
    >>> x = SimpleNamespace(text="The movie was good.")
    >>> has_person_mention(x)
    -1

    Attributes
    ----------
    name
        See above
    fault_tolerant
        See above
    """

    _nlp_config: SpacyPreprocessorConfig

    @classmethod
    def _create_or_check_preprocessor(
        cls,
        text_field: str,
        doc_field: str,
        language: str,
        disable: Optional[List[str]],
        preprocessors: List[BasePreprocessor],
        memoize: bool,
    ) -> None:
        # Create a SpacyPreprocessor if one has not yet been instantiated.
        # Otherwise, check that configuration matches already instantiated one.
        parameters = SpacyPreprocessorParameters(
            text_field=text_field,
            doc_field=doc_field,
            language=language,
            disable=disable,
            preprocessors=preprocessors,
            memoize=memoize,
        )
        if not hasattr(cls, "_nlp_config"):
            nlp = SpacyPreprocessor(**parameters._asdict())
            cls._nlp_config = SpacyPreprocessorConfig(nlp=nlp, parameters=parameters)
        elif parameters != cls._nlp_config.parameters:
            raise ValueError(
                "NLPLabelingFunction already configured with different parameters: "
                f"{cls._nlp_config.parameters}"
            )

    def __init__(
        self,
        name: str,
        f: Callable[..., int],
        resources: Optional[Mapping[str, Any]] = None,
        preprocessors: Optional[List[BasePreprocessor]] = None,
        fault_tolerant: bool = False,
        text_field: str = "text",
        doc_field: str = "doc",
        language: str = EN_CORE_WEB_SM,
        disable: Optional[List[str]] = None,
        memoize: bool = True,
    ) -> None:
        self._create_or_check_preprocessor(
            text_field, doc_field, language, disable, preprocessors or [], memoize
        )
        super().__init__(
            name,
            f,
            resources=resources,
            preprocessors=[self._nlp_config.nlp],
            fault_tolerant=fault_tolerant,
        )


class nlp_labeling_function(labeling_function):
    """Decorator to define an NLPLabelingFunction object from a function.

    Parameters
    ----------
    name
        Name of the LF
    resources
        Labeling resources passed in to ``f`` via ``kwargs``
    preprocessors
        Preprocessors to run before SpacyPreprocessor is executed
    fault_tolerant
        Output -1 if LF execution fails?
    text_field
        Name of data point text field to input
    doc_field
        Name of data point field to output parsed document to
    language
        SpaCy model to load
        See https://spacy.io/usage/models#usage
    disable
        List of pipeline components to disable
        See https://spacy.io/usage/processing-pipelines#disabling
    memoize
        Memoize preprocessor outputs?


    Example
    -------
    >>> @nlp_labeling_function()
    ... def has_person_mention(x):
    ...     person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
    ...     return 0 if len(person_ents) > 0 else -1
    >>> has_person_mention
    NLPLabelingFunction has_person_mention, Preprocessors: [SpacyPreprocessor...]

    >>> from types import SimpleNamespace
    >>> x = SimpleNamespace(text="The movie was good.")
    >>> has_person_mention(x)
    -1
    """

    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        preprocessors: Optional[List[BasePreprocessor]] = None,
        fault_tolerant: bool = False,
        text_field: str = "text",
        doc_field: str = "doc",
        language: str = EN_CORE_WEB_SM,
        disable: Optional[List[str]] = None,
        memoize: bool = True,
    ) -> None:
        super().__init__(name, resources, preprocessors, fault_tolerant)
        self.text_field = text_field
        self.doc_field = doc_field
        self.language = language
        self.disable = disable
        self.memoize = memoize

    def __call__(self, f: Callable[..., int]) -> NLPLabelingFunction:
        """Wrap a function to create an ``NLPLabelingFunction``.

        Parameters
        ----------
        f
            Function that implements the core NLP LF logic

        Returns
        -------
        NLPLabelingFunction
            New ``NLPLabelingFunction`` executing logic in wrapped function
        """
        name = self.name or f.__name__
        return NLPLabelingFunction(
            name=name,
            f=f,
            resources=self.resources,
            preprocessors=self.preprocessors,
            fault_tolerant=self.fault_tolerant,
            text_field=self.text_field,
            doc_field=self.doc_field,
            language=self.language,
            disable=self.disable,
            memoize=self.memoize,
        )
