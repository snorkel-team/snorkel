from typing import List

import spacy

from snorkel.types import FieldMap

from .core import Preprocessor

DEFAULT_DISABLE = ["parser", "ner"]
EN_CORE_WEB_SM = "en_core_web_sm"


class SpacyPreprocessor(Preprocessor):
    """Preprocessor that parses input text via a SpaCy model.

    A common approach to writing LFs over text is to first use
    a natural language parser to decompose the text into tokens,
    part-of-speech tags, etc. SpaCy (https://spacy.io/) is a
    popular tool for doing this.

    Parameters
    ----------
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
    """

    def __init__(
        self,
        text_field: str,
        doc_field: str,
        language: str = EN_CORE_WEB_SM,
        disable: List[str] = DEFAULT_DISABLE,
    ) -> None:
        name = type(self).__name__
        super().__init__(name, dict(text=text_field), dict(doc=doc_field))
        self._nlp = spacy.load(language, disable=disable)

    def run(self, text: str) -> FieldMap:  # type: ignore
        """Run the SpaCy model on input text.

        Parameters
        ----------
        text
            Text of document to parse

        Returns
        -------
        FieldMap
            Dictionary with a single key (`"doc"`), mapping to the
            parsed SpaCy `Doc` object
        """
        return dict(doc=self._nlp(text))
