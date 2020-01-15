from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.preprocess.spark import make_spark_preprocessor

from .nlp import (
    BaseNLPLabelingFunction,
    SpacyPreprocessorParameters,
    base_nlp_labeling_function,
)


class SparkNLPLabelingFunction(BaseNLPLabelingFunction):
    r"""Special labeling function type for SpaCy-based LFs running on Spark.

    This class is a Spark-compatible version of ``NLPLabelingFunction``.
    See ``NLPLabelingFunction`` for details.

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
        SpaCy model to load
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

    Attributes
    ----------
    name
        See above
    """

    @classmethod
    def _create_preprocessor(
        cls, parameters: SpacyPreprocessorParameters
    ) -> SpacyPreprocessor:
        preprocessor = SpacyPreprocessor(**parameters._asdict())
        make_spark_preprocessor(preprocessor)
        return preprocessor


class spark_nlp_labeling_function(base_nlp_labeling_function):
    """Decorator to define a SparkNLPLabelingFunction object from a function.

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
        SpaCy model to load
        See https://spacy.io/usage/models#usage
    disable
        List of pipeline components to disable
        See https://spacy.io/usage/processing-pipelines#disabling
    memoize
        Memoize preprocessor outputs?


    Example
    -------
    >>> @spark_nlp_labeling_function()
    ... def has_person_mention(x):
    ...     person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
    ...     return 0 if len(person_ents) > 0 else -1
    >>> has_person_mention
    SparkNLPLabelingFunction has_person_mention, Preprocessors: [SpacyPreprocessor...]

    >>> from pyspark.sql import Row
    >>> x = Row(text="The movie was good.")
    >>> has_person_mention(x)
    -1
    """

    _lf_cls = SparkNLPLabelingFunction
