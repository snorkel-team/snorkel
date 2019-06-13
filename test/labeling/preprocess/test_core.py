import unittest
from types import SimpleNamespace
from typing import Mapping

from snorkel.labeling.preprocess import Preprocessor, PreprocessorMode
from snorkel.types import Field


class SplitWordsPreprocessor(Preprocessor):
    def __init__(self, text_field: str, lower_field: str, words_field: str) -> None:
        super().__init__(
            dict(text=text_field), dict(lower=lower_field, words=words_field)
        )

    def preprocess(self, text: str) -> Mapping[str, Field]:  # type: ignore
        return dict(lower=text.lower(), words=text.split())


class SquarePreprocessor(Preprocessor):
    def __init__(self, x_field: str, squared_x_field: str) -> None:
        super().__init__(dict(x=x_field), dict(x=squared_x_field))

    def preprocess(self, x: float) -> Mapping[str, Field]:  # type: ignore
        return dict(x=x ** 2)


class TestPreprocessorCore(unittest.TestCase):
    def _get_example(self) -> SimpleNamespace:
        return SimpleNamespace(a=8, b="Henry has fun")

    def test_numeric_preprocessor(self) -> None:
        square = SquarePreprocessor("a", "c")
        square.set_mode(PreprocessorMode.NAMESPACE)
        example_preprocessed = square(self._get_example())
        self.assertEqual(example_preprocessed.a, 8)
        self.assertEqual(example_preprocessed.b, "Henry has fun")
        self.assertEqual(example_preprocessed.c, 64)

    def test_text_preprocessor(self) -> None:
        split_words = SplitWordsPreprocessor("b", "c", "d")
        split_words.set_mode(PreprocessorMode.NAMESPACE)
        example_preprocessed = split_words(self._get_example())
        self.assertEqual(example_preprocessed.a, 8)
        self.assertEqual(example_preprocessed.b, "Henry has fun")
        self.assertEqual(example_preprocessed.c, "henry has fun")
        self.assertEqual(example_preprocessed.d, ["Henry", "has", "fun"])

    def test_preprocessor_same_field(self) -> None:
        split_words = SplitWordsPreprocessor("b", "b", "c")
        split_words.set_mode(PreprocessorMode.NAMESPACE)
        example_preprocessed = split_words(self._get_example())
        self.assertEqual(example_preprocessed.a, 8)
        self.assertEqual(example_preprocessed.b, "henry has fun")
        self.assertEqual(example_preprocessed.c, ["Henry", "has", "fun"])
