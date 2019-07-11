import unittest
from typing import Optional

import pytest
from pyspark.sql import Row

from snorkel.map import Mapper, lambda_mapper
from snorkel.map.spark import make_spark_mapper
from snorkel.types import DataPoint, FieldMap


class SplitWordsMapper(Mapper):
    def __init__(
        self, name: str, text_field: str, lower_field: str, words_field: str
    ) -> None:
        super().__init__(
            name, dict(text=text_field), dict(lower=lower_field, words=words_field)
        )

    def run(self, text: str) -> FieldMap:  # type: ignore
        return dict(lower=text.lower(), words=text.split())


class SplitWordsMapperDefaultArgs(Mapper):
    def run(self, text: str) -> FieldMap:  # type: ignore
        return dict(lower=text.lower(), words=text.split())


class MapperReturnsNone(Mapper):
    def run(self, text: str) -> Optional[FieldMap]:  # type: ignore
        return None


class SquareHitTracker:
    def __init__(self):
        self.n_hits = 0

    def __call__(self, x: float) -> float:
        self.n_hits += 1
        return x ** 2


@lambda_mapper()
def square(x: DataPoint) -> DataPoint:
    fields = x.asDict()
    fields["num_squared"] = x.num ** 2
    return Row(**fields)


@lambda_mapper()
def modify_in_place(x: DataPoint) -> DataPoint:
    x.d["my_key"] = 0
    return Row(num=x.num, d=x.d, d_new=x.d)


class TestMapperCore(unittest.TestCase):
    def _get_x(self, num=8, text="Henry has fun") -> Row:
        return Row(num=num, text=text)

    def _get_x_dict(self) -> Row:
        return Row(num=8, d=dict(my_key=1))

    @pytest.mark.spark
    def test_numeric_mapper(self) -> None:
        x_mapped = square(self._get_x())
        # NB: not using `self.assertIsNotNone` due to mypy
        # See https://github.com/python/mypy/issues/5088
        assert x_mapped is not None
        self.assertEqual(x_mapped.num, 8)
        self.assertEqual(x_mapped.text, "Henry has fun")
        self.assertEqual(x_mapped.num_squared, 64)

    @pytest.mark.spark
    def test_text_mapper(self) -> None:
        split_words = SplitWordsMapper(
            "split_words", "text", "text_lower", "text_words"
        )
        split_words_spark = make_spark_mapper(split_words)
        x_mapped = split_words_spark(self._get_x())
        assert x_mapped is not None
        self.assertEqual(x_mapped.num, 8)
        self.assertEqual(x_mapped.text, "Henry has fun")
        self.assertEqual(x_mapped.text_lower, "henry has fun")
        self.assertEqual(x_mapped.text_words, ["Henry", "has", "fun"])

    @pytest.mark.spark
    def test_mapper_same_field(self) -> None:
        split_words = SplitWordsMapper("split_words", "text", "text", "text_words")
        split_words_spark = make_spark_mapper(split_words)
        x = self._get_x()
        x_mapped = split_words_spark(x)
        self.assertEqual(x.num, 8)
        self.assertEqual(x.text, "Henry has fun")
        self.assertFalse(hasattr(x, "text_words"))
        assert x_mapped is not None
        self.assertEqual(x_mapped.num, 8)
        self.assertEqual(x_mapped.text, "henry has fun")
        self.assertEqual(x_mapped.text_words, ["Henry", "has", "fun"])

    @pytest.mark.spark
    def test_mapper_default_args(self) -> None:
        split_words = SplitWordsMapperDefaultArgs("split_words")
        split_words_spark = make_spark_mapper(split_words)
        x_mapped = split_words_spark(self._get_x())
        assert x_mapped is not None
        self.assertEqual(x_mapped.num, 8)
        self.assertEqual(x_mapped.text, "Henry has fun")
        self.assertEqual(x_mapped.lower, "henry has fun")
        self.assertEqual(x_mapped.words, ["Henry", "has", "fun"])

    @pytest.mark.spark
    def test_mapper_in_place(self) -> None:
        x = self._get_x_dict()
        x_mapped = modify_in_place(x)
        self.assertEqual(x.num, 8)
        self.assertEqual(x.d, dict(my_key=1))
        self.assertFalse(hasattr(x, "d_new"))
        assert x_mapped is not None
        self.assertEqual(x_mapped.num, 8)
        self.assertEqual(x_mapped.d, dict(my_key=0))
        self.assertEqual(x_mapped.d_new, dict(my_key=0))

    @pytest.mark.spark
    def test_mapper_returns_none(self) -> None:
        mapper = MapperReturnsNone("none_mapper")
        mapper_spark = make_spark_mapper(mapper)
        x_mapped = mapper_spark(self._get_x())
        self.assertIsNone(x_mapped)

    @pytest.mark.spark
    def test_decorator_mapper_memoized(self) -> None:
        square_hit_tracker = SquareHitTracker()

        @lambda_mapper(memoize=True)
        def square(x: DataPoint) -> DataPoint:
            fields = x.asDict()
            fields["num_squared"] = square_hit_tracker(x.num)
            return Row(**fields)

        x8 = self._get_x()
        x9 = self._get_x(9)
        x8_mapped = square(x8)
        assert x8_mapped is not None
        self.assertEqual(x8_mapped.num_squared, 64)
        self.assertEqual(square_hit_tracker.n_hits, 1)
        x8_mapped = square(x8)
        assert x8_mapped is not None
        self.assertEqual(x8_mapped.num_squared, 64)
        self.assertEqual(square_hit_tracker.n_hits, 1)
        x19_mapped = square(x9)
        assert x19_mapped is not None
        self.assertEqual(x19_mapped.num_squared, 81)
        self.assertEqual(square_hit_tracker.n_hits, 2)
        x8_mapped = square(x8)
        assert x8_mapped is not None
        self.assertEqual(x8_mapped.num_squared, 64)
        self.assertEqual(square_hit_tracker.n_hits, 2)

        square.reset_cache()
        x8_mapped = square(x8)
        assert x8_mapped is not None
        self.assertEqual(x8_mapped.num_squared, 64)
        self.assertEqual(square_hit_tracker.n_hits, 3)

    @pytest.mark.spark
    def test_decorator_mapper_memoized_none(self) -> None:
        square_hit_tracker = SquareHitTracker()

        @lambda_mapper(memoize=True)
        def square(x: DataPoint) -> DataPoint:
            fields = x.asDict()
            fields["num_squared"] = square_hit_tracker(x.num)
            if x.num == 21:
                return None
            return Row(**fields)

        x21 = self._get_x(21)
        x21_mapped = square(x21)
        self.assertIsNone(x21_mapped)
        self.assertEqual(square_hit_tracker.n_hits, 1)
        x21_mapped = square(x21)
        self.assertIsNone(x21_mapped)
        self.assertEqual(square_hit_tracker.n_hits, 1)

    @pytest.mark.spark
    def test_decorator_mapper_not_memoized(self) -> None:
        square_hit_tracker = SquareHitTracker()

        @lambda_mapper(memoize=False)
        def square(x: DataPoint) -> DataPoint:
            fields = x.asDict()
            fields["num_squared"] = square_hit_tracker(x.num)
            return Row(**fields)

        x8 = self._get_x()
        x9 = self._get_x(9)
        x8_mapped = square(x8)
        assert x8_mapped is not None
        self.assertEqual(x8_mapped.num_squared, 64)
        self.assertEqual(square_hit_tracker.n_hits, 1)
        x8_mapped = square(x8)
        assert x8_mapped is not None
        self.assertEqual(x8_mapped.num_squared, 64)
        self.assertEqual(square_hit_tracker.n_hits, 2)
        x19_mapped = square(x9)
        assert x19_mapped is not None
        self.assertEqual(x19_mapped.num_squared, 81)
        self.assertEqual(square_hit_tracker.n_hits, 3)
        x8_mapped = square(x8)
        assert x8_mapped is not None
        self.assertEqual(x8_mapped.num_squared, 64)
        self.assertEqual(square_hit_tracker.n_hits, 4)
