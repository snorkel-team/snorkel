import unittest
from types import SimpleNamespace

from snorkel.labeling.preprocess import Preprocessor, PreprocessorMode, preprocessor
from snorkel.types import FieldMap


class SplitWordsPreprocessor(Preprocessor):
    def __init__(self, text_field: str, lower_field: str, words_field: str) -> None:
        super().__init__(
            dict(text=text_field), dict(lower=lower_field, words=words_field)
        )

    def preprocess(self, text: str) -> FieldMap:  # type: ignore
        return dict(lower=text.lower(), words=text.split())


@preprocessor
def square(num: float) -> FieldMap:
    return dict(num_squared=num ** 2)


class TestPreprocessorCore(unittest.TestCase):
    def _get_x(self) -> SimpleNamespace:
        return SimpleNamespace(num=8, text="Henry has fun")

    def test_numeric_preprocessor(self) -> None:
        square.set_mode(PreprocessorMode.NAMESPACE)
        x_preprocessed = square(self._get_x())
        self.assertEqual(x_preprocessed.num, 8)
        self.assertEqual(x_preprocessed.text, "Henry has fun")
        self.assertEqual(x_preprocessed.num_squared, 64)

    def test_text_preprocessor(self) -> None:
        split_words = SplitWordsPreprocessor("text", "text_lower", "text_words")
        split_words.set_mode(PreprocessorMode.NAMESPACE)
        x_preprocessed = split_words(self._get_x())
        self.assertEqual(x_preprocessed.num, 8)
        self.assertEqual(x_preprocessed.text, "Henry has fun")
        self.assertEqual(x_preprocessed.text_lower, "henry has fun")
        self.assertEqual(x_preprocessed.text_words, ["Henry", "has", "fun"])

    def test_preprocessor_same_field(self) -> None:
        split_words = SplitWordsPreprocessor("text", "text", "text_words")
        split_words.set_mode(PreprocessorMode.NAMESPACE)
        x_preprocessed = split_words(self._get_x())
        self.assertEqual(x_preprocessed.num, 8)
        self.assertEqual(x_preprocessed.text, "henry has fun")
        self.assertEqual(x_preprocessed.text_words, ["Henry", "has", "fun"])

    def test_preprocessor_mode(self) -> None:
        x = self._get_x()

        square.set_mode(18)  # type: ignore
        with self.assertRaises(ValueError):
            square(x)

        square.set_mode(PreprocessorMode.NONE)
        with self.assertRaises(ValueError):
            square(x)

        square.set_mode(PreprocessorMode.DASK)
        with self.assertRaises(NotImplementedError):
            square(x)

        square.set_mode(PreprocessorMode.SPARK)
        with self.assertRaises(NotImplementedError):
            square(x)
