import unittest
from types import SimpleNamespace

from snorkel.map import Mapper, MapperMode, lambda_mapper
from snorkel.types import FieldMap


class SplitWordsMapper(Mapper):
    def __init__(self, text_field: str, lower_field: str, words_field: str) -> None:
        super().__init__(
            dict(text=text_field), dict(lower=lower_field, words=words_field)
        )

    def run(self, text: str) -> FieldMap:  # type: ignore
        return dict(lower=text.lower(), words=text.split())


class SplitWordsMapperDefaultArgs(Mapper):
    def run(self, text: str) -> FieldMap:  # type: ignore
        return dict(lower=text.lower(), words=text.split())


@lambda_mapper
def square(num: float) -> FieldMap:
    return dict(num_squared=num ** 2)


class TestMapperCore(unittest.TestCase):
    def _get_x(self) -> SimpleNamespace:
        return SimpleNamespace(num=8, text="Henry has fun")

    def test_numeric_mapper(self) -> None:
        square.set_mode(MapperMode.NAMESPACE)
        x_mapped = square(self._get_x())
        self.assertEqual(x_mapped.num, 8)
        self.assertEqual(x_mapped.text, "Henry has fun")
        self.assertEqual(x_mapped.num_squared, 64)

    def test_text_mapper(self) -> None:
        split_words = SplitWordsMapper("text", "text_lower", "text_words")
        split_words.set_mode(MapperMode.NAMESPACE)
        x_mapped = split_words(self._get_x())
        self.assertEqual(x_mapped.num, 8)
        self.assertEqual(x_mapped.text, "Henry has fun")
        self.assertEqual(x_mapped.text_lower, "henry has fun")
        self.assertEqual(x_mapped.text_words, ["Henry", "has", "fun"])

    def test_mapper_same_field(self) -> None:
        split_words = SplitWordsMapper("text", "text", "text_words")
        split_words.set_mode(MapperMode.NAMESPACE)
        x_mapped = split_words(self._get_x())
        self.assertEqual(x_mapped.num, 8)
        self.assertEqual(x_mapped.text, "henry has fun")
        self.assertEqual(x_mapped.text_words, ["Henry", "has", "fun"])

    def test_mapper_default_args(self) -> None:
        split_words = SplitWordsMapperDefaultArgs()
        split_words.set_mode(MapperMode.NAMESPACE)
        x_mapped = split_words(self._get_x())
        self.assertEqual(x_mapped.num, 8)
        self.assertEqual(x_mapped.text, "Henry has fun")
        self.assertEqual(x_mapped.lower, "henry has fun")
        self.assertEqual(x_mapped.words, ["Henry", "has", "fun"])

    def test_mapper_mode(self) -> None:
        x = self._get_x()

        square.set_mode(18)  # type: ignore
        with self.assertRaises(ValueError):
            square(x)

        square.set_mode(MapperMode.NONE)
        with self.assertRaises(ValueError):
            square(x)

        square.set_mode(MapperMode.DASK)
        with self.assertRaises(NotImplementedError):
            square(x)

        square.set_mode(MapperMode.SPARK)
        with self.assertRaises(NotImplementedError):
            square(x)
