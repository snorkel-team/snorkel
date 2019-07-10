from snorkel.map import BaseMapper, LambdaMapper, Mapper, lambda_mapper

"""Base classes for preprocessors.

A preprocessor is a data point to data point mapping in a labeling
pipeline. This allows LFs to share common preprocessing steps that
make it easier to express labeling logic. A simple example for text
processing is concatenating the title and body of an article. For a
more complex example, see `snorkel.labeling.preprocess.nlp.SpacyPreprocessor`.
"""


BasePreprocessor = BaseMapper


class Preprocessor(Mapper):
    """Base class for preprocessors.

    See `snorkel.map.core.Mapper` for details.
    """

    pass


class LambdaPreprocessor(LambdaMapper):
    """Convenience class for definining preprocessors from functions.

    See `snorkel.map.core.LambdaMapper` for details.
    """

    pass


class preprocessor(lambda_mapper):
    """Decorate functions to create preprocessors.

    See `snorkel.map.core.lambda_mapper` for details.

    Example
    -------

    ```
    @preprocessor()
    def combine_text_preprocessor(x: DataPoint) -> DataPoint:
        x.article = f"{x.title} {x.body}"
        return x

    spacy_preprocessor = SpacyPreprocessor("article", "article_parsed")

    preprocessors = [combine_text_preprocessor, spacy_preprocessor]
    @labeling_function(preprocessors=preprocessors)
    def article_mentions_person(x: DataPoint) -> int:
        for ent in x.article_parsed.ents:
            if ent.label_ == "PERSON":
                return ABSTAIN
        return NEGATIVE
    ```
    """

    pass
