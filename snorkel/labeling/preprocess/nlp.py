from typing import List

import spacy

from snorkel.types import FieldMap

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

    def run(self, text: str) -> FieldMap:  # type: ignore
        return dict(doc=self._nlp(text))
