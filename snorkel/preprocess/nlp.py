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
    popular tool for doing this. This preprocessor adds a
    SpaCy ``Doc`` object to the data point. A ``Doc`` object is
    a sequence of ``Token`` objects, which contain information
    on lemmatization, parts-of-speech, etc. ``Doc`` objects also
    contain fields like ``Doc.ents``, a list of named entities,
    and ``Doc.noun_chunks``, a list of noun phrases. For details
    of SpaCy ``Doc`` objects and a full attribute listing,
    see https://spacy.io/api/doc.

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
    pre
        Preprocessors to run before this preprocessor is executed
    memoize
        Memoize preprocessor outputs?
    gpu
        Prefer Spacy GPU processing?
    """

    def __init__(
        self,
        text_field: str,
        doc_field: str,
        language: str = EN_CORE_WEB_SM,
        disable: Optional[List[str]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        memoize: bool = False,
        gpu: bool = False,
    ) -> None:
        name = type(self).__name__
        super().__init__(
            name,
            field_names=dict(text=text_field),
            mapped_field_names=dict(doc=doc_field),
            pre=pre,
            memoize=memoize,
        )
        self.gpu = gpu
        if self.gpu:
            spacy.prefer_gpu()
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
