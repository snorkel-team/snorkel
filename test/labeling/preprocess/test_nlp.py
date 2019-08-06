import unittest
from types import SimpleNamespace

from snorkel.preprocess.nlp import SpacyPreprocessor


class TestSpacyPreprocessor(unittest.TestCase):
    def test_spacy_preprocessor(self) -> None:
        x = SimpleNamespace(text="Jane plays soccer.")
        preprocessor = SpacyPreprocessor("text", "doc")
        x_preprocessed = preprocessor(x)
        assert x_preprocessed is not None
        self.assertEqual(len(x_preprocessed.doc), 4)
        token = x_preprocessed.doc[0]
        self.assertEqual(token.text, "Jane")
        self.assertEqual(token.pos_, "PROPN")
