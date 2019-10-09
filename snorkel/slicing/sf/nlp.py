from snorkel.labeling.lf.nlp import (
    BaseNLPLabelingFunction,
    SpacyPreprocessorParameters,
    base_nlp_labeling_function,
)
from snorkel.preprocess.nlp import SpacyPreprocessor


class NLPSlicingFunction(BaseNLPLabelingFunction):
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

    Raises
    ------
    ValueError
        Calling incorrectly defined preprocessors

    Example
    -------
    >>> def f(x):
    ...     person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
    ...     return len(person_ents) > 0
    >>> has_person_mention = NLPSlicingFunction(name="has_person_mention", f=f)
    >>> has_person_mention
    NLPSlicingFunction has_person_mention, Preprocessors: [SpacyPreprocessor...]

    >>> from types import SimpleNamespace
    >>> x = SimpleNamespace(text="The movie was good.")
    >>> has_person_mention(x)
    False

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


class nlp_slicing_function(base_nlp_labeling_function):
    """Decorator to define a NLPSlicingFunction child object from a function.

    TODO: Implement a common parent decorator for Snorkel operators
    """

    _lf_cls = NLPSlicingFunction
