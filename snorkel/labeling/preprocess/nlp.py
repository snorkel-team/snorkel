from typing import List, Optional

import spacy

from snorkel.types import FieldMap

from .core import BasePreprocessor, Preprocessor

EN_CORE_WEB_SM = "en_core_web_sm"


class SpacyPreprocessor(Preprocessor):
    """Preprocessor that parses input text via a SpaCy model.

    A common approach to writing LFs over text is to first use
    a natural language parser to decompose the text into tokens,
    part-of-speech tags, etc. SpaCy (https://spacy.io/) is a
    popular tool for doing this. For details of SpaCy ``Doc``objects
    and a full attribute listing, see https://spacy.io/api/doc.

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
    preprocessors
        Preprocessors to run before this preprocessor is executed
    memoize
        Memoize preprocessor outputs?
    """

    def __init__(
        self,
        text_field: str,
        doc_field: str,
        language: str = EN_CORE_WEB_SM,
        disable: Optional[List[str]] = None,
        preprocessors: Optional[List[BasePreprocessor]] = None,
        memoize: bool = False,
    ) -> None:
        name = type(self).__name__
        super().__init__(
            name,
            field_names=dict(text=text_field),
            mapped_field_names=dict(doc=doc_field),
            pre=preprocessors,
            memoize=memoize,
        )
        self._nlp = spacy.load(language, disable=disable or [])

    def run(self, text: str) -> FieldMap:  # type: ignore
        """Run the SpaCy model on input text.

        Parameters
        ----------
        text
            Text of document to parse

        Returns
        -------
        FieldMap
            Dictionary with a single key (``"doc"``), mapping to the
            parsed SpaCy ``Doc`` object
        """
        # Note: not trying to add the fields of `Doc` to top-level
        # as most are Cython property methods computed on the fly.
        return dict(doc=self._nlp(text))
