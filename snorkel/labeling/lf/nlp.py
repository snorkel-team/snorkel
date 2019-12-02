from typing import Any, Callable, List, Mapping, NamedTuple, Optional

from snorkel.preprocess import BasePreprocessor
from snorkel.preprocess.nlp import EN_CORE_WEB_SM, SpacyPreprocessor

from .core import LabelingFunction, labeling_function


class SpacyPreprocessorParameters(NamedTuple):
    """Parameters needed to construct a SpacyPreprocessor.

    See ``snorkel.preprocess.nlp.SpacyPreprocessor``.
    """

    text_field: str
    doc_field: str
    language: str
    disable: Optional[List[str]]
    pre: List[BasePreprocessor]
    memoize: bool
    gpu: bool


class SpacyPreprocessorConfig(NamedTuple):
    """Tuple of SpacyPreprocessor and the parameters used to construct it."""

    nlp: SpacyPreprocessor
    parameters: SpacyPreprocessorParameters


class BaseNLPLabelingFunction(LabelingFunction):
    """Base class for spaCy-based LFs."""

    _nlp_config: SpacyPreprocessorConfig

    @classmethod
    def _create_preprocessor(
        cls, parameters: SpacyPreprocessorParameters
    ) -> SpacyPreprocessor:
        raise NotImplementedError

    @classmethod
    def _create_or_check_preprocessor(
        cls,
        text_field: str,
        doc_field: str,
        language: str,
        disable: Optional[List[str]],
        pre: List[BasePreprocessor],
        memoize: bool,
        gpu: bool,
    ) -> None:
        # Create a SpacyPreprocessor if one has not yet been instantiated.
        # Otherwise, check that configuration matches already instantiated one.
        parameters = SpacyPreprocessorParameters(
            text_field=text_field,
            doc_field=doc_field,
            language=language,
            disable=disable,
            pre=pre,
            memoize=memoize,
            gpu=gpu,
        )
        if not hasattr(cls, "_nlp_config"):
            nlp = cls._create_preprocessor(parameters)
            cls._nlp_config = SpacyPreprocessorConfig(nlp=nlp, parameters=parameters)
        elif parameters != cls._nlp_config.parameters:
            raise ValueError(
                f"{cls.__name__} already configured with different parameters: "
                f"{cls._nlp_config.parameters}"
            )

    def __init__(
        self,
        name: str,
        f: Callable[..., int],
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        text_field: str = "text",
        doc_field: str = "doc",
        language: str = EN_CORE_WEB_SM,
        disable: Optional[List[str]] = None,
        memoize: bool = True,
        gpu: bool = False,
    ) -> None:
        self._create_or_check_preprocessor(
            text_field, doc_field, language, disable, pre or [], memoize, gpu
        )
        super().__init__(name, f, resources=resources, pre=[self._nlp_config.nlp])


class NLPLabelingFunction(BaseNLPLabelingFunction):
    r"""Special labeling function type for spaCy-based LFs.

    This class is a special version of ``LabelingFunction``. It
    has a ``SpacyPreprocessor`` integrated which shares a cache
    with all other ``NLPLabelingFunction`` instances. This makes
    it easy to define LFs that have a text input field and have
    logic written over spaCy ``Doc`` objects. Examples passed
    into an ``NLPLabelingFunction`` will have a new field which
    can be accessed which contains a spaCy ``Doc``. By default,
    this field is called ``doc``. A ``Doc`` object is
    a sequence of ``Token`` objects, which contain information
    on lemmatization, parts-of-speech, etc. ``Doc`` objects also
    contain fields like ``Doc.ents``, a list of named entities,
    and ``Doc.noun_chunks``, a list of noun phrases. For details
    of spaCy ``Doc`` objects and a full attribute listing,
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
    pre
        Preprocessors to run before SpacyPreprocessor is executed
    text_field
        Name of data point text field to input
    doc_field
        Name of data point field to output parsed document to
    language
        spaCy model to load
        See https://spacy.io/usage/models#usage
    disable
        List of pipeline components to disable
        See https://spacy.io/usage/processing-pipelines#disabling
    memoize
        Memoize preprocessor outputs?
    gpu
        Prefer Spacy GPU processing?

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
    """

    @classmethod
    def _create_preprocessor(
        cls, parameters: SpacyPreprocessorParameters
    ) -> SpacyPreprocessor:
        return SpacyPreprocessor(**parameters._asdict())


class base_nlp_labeling_function(labeling_function):
    """Decorator to define a BaseNLPLabelingFunction child object from a function."""

    _lf_cls: Optional[type] = None

    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        text_field: str = "text",
        doc_field: str = "doc",
        language: str = EN_CORE_WEB_SM,
        disable: Optional[List[str]] = None,
        memoize: bool = True,
        gpu: bool = False,
    ) -> None:
        super().__init__(name, resources, pre)
        self.text_field = text_field
        self.doc_field = doc_field
        self.language = language
        self.disable = disable
        self.memoize = memoize
        self.gpu = gpu

    def __call__(self, f: Callable[..., int]) -> BaseNLPLabelingFunction:
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
        if self._lf_cls is None:
            raise NotImplementedError("_lf_cls must be defined")
        name = self.name or f.__name__
        return self._lf_cls(
            name=name,
            f=f,
            resources=self.resources,
            pre=self.pre,
            text_field=self.text_field,
            doc_field=self.doc_field,
            language=self.language,
            disable=self.disable,
            memoize=self.memoize,
            gpu=self.gpu,
        )


class nlp_labeling_function(base_nlp_labeling_function):
    """Decorator to define an NLPLabelingFunction object from a function.

    Parameters
    ----------
    name
        Name of the LF
    resources
        Labeling resources passed in to ``f`` via ``kwargs``
    pre
        Preprocessors to run before SpacyPreprocessor is executed
    text_field
        Name of data point text field to input
    doc_field
        Name of data point field to output parsed document to
    language
        spaCy model to load
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

    _lf_cls = NLPLabelingFunction
