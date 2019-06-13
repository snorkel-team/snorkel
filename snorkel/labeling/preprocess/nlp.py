from typing import List, Mapping

import spacy

from snorkel.types import Field

from .core import Preprocessor

DEFAULT_DISABLE = ["parser", "ner"]
EN_CORE_WEB_SM = "en_core_web_sm"


class SpacyPreprocessor(Preprocessor):
    def __init__(
        self,
        text_field: str,
        doc_field: str,
        language: str = EN_CORE_WEB_SM,
        disable: List[str] = DEFAULT_DISABLE,
    ) -> None:
        super().__init__(dict(text=text_field), dict(doc=doc_field))
        self._nlp = spacy.load(language, disable=disable)

    def preprocess(self, text: str) -> Mapping[str, Field]:  # type: ignore
        return dict(doc=self._nlp(text))
