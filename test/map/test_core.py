import unittest
from types import SimpleNamespace
from typing import Any, Optional

from snorkel.map import Mapper, MapperMode, lambda_mapper
from snorkel.types import DataPoint, FieldMap


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


class MapperReturnsNone(Mapper):
    def run(self, text: str) -> Optional[FieldMap]:  # type: ignore
        return None


class MapperWithArgs(Mapper):
    def run(self, text: str, *args: Any) -> Optional[FieldMap]:  # type: ignore
        return None


class MapperWithKwargs(Mapper):
    def run(self, text: str, **kwargs: Any) -> Optional[FieldMap]:  # type: ignore
        return None


@lambda_mapper
def square(x: DataPoint) -> DataPoint:
    x.num_squared = x.num ** 2
    return x


@lambda_mapper
def modify_in_place(x: DataPoint) -> DataPoint:
    x.d["my_key"] = 0
    x.d_new = x.d
    return x


class TestMapperCore(unittest.TestCase):
    def _get_x(self) -> SimpleNamespace:
        return SimpleNamespace(num=8, text="Henry has fun")

    def _get_x_dict(self) -> SimpleNamespace:
        return SimpleNamespace(num=8, d=dict(my_key=1))

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
        x = self._get_x()
        x_mapped = split_words(x)
        self.assertEqual(x.num, 8)
        self.assertEqual(x.text, "Henry has fun")
        self.assertFalse(hasattr(x, "text_words"))
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

    def test_mapper_in_place(self) -> None:
        modify_in_place.set_mode(MapperMode.NAMESPACE)
        x = self._get_x_dict()
        x_mapped = modify_in_place(x)
        self.assertEqual(x.num, 8)
        self.assertEqual(x.d, dict(my_key=1))
        self.assertFalse(hasattr(x, "d_new"))
        self.assertEqual(x_mapped.num, 8)
        self.assertEqual(x_mapped.d, dict(my_key=0))
        self.assertEqual(x_mapped.d_new, dict(my_key=0))

    def test_mapper_returns_none(self) -> None:
        mapper = MapperReturnsNone()
        mapper.set_mode(MapperMode.NAMESPACE)
        x_mapped = mapper(self._get_x())
        self.assertIsNone(x_mapped)

    def test_mapper_with_args_kwargs(self) -> None:
        with self.assertRaises(ValueError):
            MapperWithArgs()

        with self.assertRaises(ValueError):
            MapperWithKwargs()

    def test_mapper_mode(self) -> None:
        x = self._get_x()
        split_words = SplitWordsMapper("text", "text_lower", "text_words")

        split_words.set_mode(18)  # type: ignore
        with self.assertRaises(ValueError):
            split_words(x)

        split_words.set_mode(MapperMode.NONE)
        with self.assertRaises(ValueError):
            split_words(x)

        split_words.set_mode(MapperMode.DASK)
        with self.assertRaises(NotImplementedError):
            split_words(x)

        split_words.set_mode(MapperMode.SPARK)
        with self.assertRaises(NotImplementedError):
            split_words(x)
