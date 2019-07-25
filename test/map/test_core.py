import unittest
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd
import spacy

from snorkel.map import Mapper, lambda_mapper
from snorkel.map.core import get_hashable
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


class MapperWithArgs(Mapper):
    def run(self, text: str, *args: Any) -> Optional[FieldMap]:  # type: ignore
        return None


class MapperWithKwargs(Mapper):
    def run(self, text: str, **kwargs: Any) -> Optional[FieldMap]:  # type: ignore
        return None


class MapperWithPre(Mapper):
    def run(self, num_squared: float) -> Optional[FieldMap]:  # type: ignore
        return dict(double_num_squared=2 * num_squared)


class MapperWithPre2(Mapper):
    def run(self, double_num_squared: float) -> Optional[FieldMap]:  # type: ignore
        return dict(double_num_squared_plus_1=double_num_squared + 1)


class SquareHitTracker:
    def __init__(self):
        self.n_hits = 0

    def __call__(self, x: float) -> float:
        self.n_hits += 1
        return x ** 2


@lambda_mapper()
def square(x: DataPoint) -> DataPoint:
    x.num_squared = x.num ** 2
    return x


@lambda_mapper()
def modify_in_place(x: DataPoint) -> DataPoint:
    x.d["my_key"] = 0
    x.d_new = x.d
    return x


class TestMapperCore(unittest.TestCase):
    def _get_x(self, num=8, text="Henry has fun") -> SimpleNamespace:
        return SimpleNamespace(num=num, text=text)

    def _get_x_dict(self) -> SimpleNamespace:
        return SimpleNamespace(num=8, d=dict(my_key=1))

    def test_numeric_mapper(self) -> None:
        x_mapped = square(self._get_x())
        # NB: not using `self.assertIsNotNone` due to mypy
        # See https://github.com/python/mypy/issues/5088
        assert x_mapped is not None
        x_expected = SimpleNamespace(num=8, text="Henry has fun", num_squared=64)
        self.assertEqual(x_mapped, x_expected)

    def test_text_mapper(self) -> None:
        split_words = SplitWordsMapper(
            "split_words", "text", "text_lower", "text_words"
        )
        x_mapped = split_words(self._get_x())
        assert x_mapped is not None
        x_expected = SimpleNamespace(
            num=8,
            text="Henry has fun",
            text_lower="henry has fun",
            text_words=["Henry", "has", "fun"],
        )
        self.assertEqual(x_mapped, x_expected)

    def test_mapper_same_field(self) -> None:
        split_words = SplitWordsMapper("split_words", "text", "text", "text_words")
        x = self._get_x()
        x_mapped = split_words(x)
        self.assertEqual(x, self._get_x())
        assert x_mapped is not None
        x_expected = SimpleNamespace(
            num=8, text="henry has fun", text_words=["Henry", "has", "fun"]
        )
        self.assertEqual(x_mapped, x_expected)

    def test_mapper_default_args(self) -> None:
        split_words = SplitWordsMapperDefaultArgs("split_words")
        x_mapped = split_words(self._get_x())
        assert x_mapped is not None
        x_expected = SimpleNamespace(
            num=8,
            text="Henry has fun",
            lower="henry has fun",
            words=["Henry", "has", "fun"],
        )
        self.assertEqual(x_mapped, x_expected)

    def test_mapper_in_place(self) -> None:
        x = self._get_x_dict()
        x_mapped = modify_in_place(x)
        self.assertEqual(x, self._get_x_dict())
        assert x_mapped is not None
        x_expected = SimpleNamespace(num=8, d=dict(my_key=0), d_new=dict(my_key=0))
        self.assertEqual(x_mapped, x_expected)

    def test_mapper_returns_none(self) -> None:
        mapper = MapperReturnsNone("none_mapper")
        x_mapped = mapper(self._get_x())
        self.assertIsNone(x_mapped)

    def test_mapper_pre(self) -> None:
        mapper_no_pre = MapperWithPre("pre_mapper")
        x = self._get_x(3)
        with self.assertRaises(AttributeError):
            x_mapped = mapper_no_pre(x)

        mapper_pre = MapperWithPre("pre_mapper", pre=[square])
        x = self._get_x(3)
        x_mapped = mapper_pre(x)
        self.assertEqual(x, self._get_x(3))
        assert x_mapped is not None
        x_expected = SimpleNamespace(
            num=3, num_squared=9, double_num_squared=18, text="Henry has fun"
        )
        self.assertEqual(x_mapped, x_expected)

        mapper_pre_2 = MapperWithPre2("pre_mapper", pre=[mapper_pre])
        x = self._get_x(3)
        x_mapped = mapper_pre_2(x)
        self.assertEqual(x, self._get_x(3))
        assert x_mapped is not None
        x_expected = SimpleNamespace(
            num=3,
            num_squared=9,
            double_num_squared=18,
            double_num_squared_plus_1=19,
            text="Henry has fun",
        )
        self.assertEqual(x_mapped, x_expected)

    def test_mapper_pre_decorator(self) -> None:
        @lambda_mapper()
        def mapper_no_pre(x: DataPoint) -> DataPoint:
            x.double_num_squared = 2 * x.num_squared
            return x

        x = self._get_x(3)
        with self.assertRaises(AttributeError):
            x_mapped = mapper_no_pre(x)

        @lambda_mapper(pre=[square])
        def mapper_pre(x: DataPoint) -> DataPoint:
            x.double_num_squared = 2 * x.num_squared
            return x

        x = self._get_x(3)
        x_mapped = mapper_pre(x)
        self.assertEqual(x, self._get_x(3))
        assert x_mapped is not None
        x_expected = SimpleNamespace(
            num=3, num_squared=9, double_num_squared=18, text="Henry has fun"
        )
        self.assertEqual(x_mapped, x_expected)

        @lambda_mapper(pre=[mapper_pre])
        def mapper_pre_2(x: DataPoint) -> DataPoint:
            x.double_num_squared_plus_1 = x.double_num_squared + 1
            return x

        x = self._get_x(3)
        x_mapped = mapper_pre_2(x)
        self.assertEqual(x, self._get_x(3))
        assert x_mapped is not None
        x_expected = SimpleNamespace(
            num=3,
            num_squared=9,
            double_num_squared=18,
            double_num_squared_plus_1=19,
            text="Henry has fun",
        )
        self.assertEqual(x_mapped, x_expected)

    def test_decorator_mapper_memoized(self) -> None:
        square_hit_tracker = SquareHitTracker()

        @lambda_mapper(memoize=True)
        def square(x: DataPoint) -> DataPoint:
            x.num_squared = square_hit_tracker(x.num)
            return x

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
        x9_mapped = square(x9)
        assert x9_mapped is not None
        self.assertEqual(x9_mapped.num_squared, 81)
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

    def test_decorator_mapper_memoized_none(self) -> None:
        square_hit_tracker = SquareHitTracker()

        @lambda_mapper(memoize=True)
        def square(x: DataPoint) -> DataPoint:
            x.num_squared = square_hit_tracker(x.num)
            if x.num == 21:
                return None
            return x

        x21 = self._get_x(21)
        x21_mapped = square(x21)
        self.assertIsNone(x21_mapped)
        self.assertEqual(square_hit_tracker.n_hits, 1)
        x21_mapped = square(x21)
        self.assertIsNone(x21_mapped)
        self.assertEqual(square_hit_tracker.n_hits, 1)

    def test_decorator_mapper_not_memoized(self) -> None:
        square_hit_tracker = SquareHitTracker()

        @lambda_mapper(memoize=False)
        def square(x: DataPoint) -> DataPoint:
            x.num_squared = square_hit_tracker(x.num)
            return x

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

    def test_mapper_pre_memoized(self) -> None:
        square_hit_tracker = SquareHitTracker()

        @lambda_mapper(memoize=False)
        def square(x: DataPoint) -> DataPoint:
            x.num_squared = square_hit_tracker(x.num)
            return x

        @lambda_mapper(pre=[square], memoize=True)
        def mapper_pre(x: DataPoint) -> DataPoint:
            x.double_num_squared = 2 * x.num_squared
            return x

        x8 = self._get_x()
        x9 = self._get_x(9)
        x8_mapped = mapper_pre(x8)
        assert x8_mapped is not None
        self.assertEqual(x8_mapped.double_num_squared, 128)
        self.assertEqual(square_hit_tracker.n_hits, 1)
        x8_mapped = mapper_pre(x8)
        assert x8_mapped is not None
        self.assertEqual(x8_mapped.double_num_squared, 128)
        self.assertEqual(square_hit_tracker.n_hits, 1)
        x9_mapped = mapper_pre(x9)
        assert x9_mapped is not None
        self.assertEqual(x9_mapped.double_num_squared, 162)
        self.assertEqual(square_hit_tracker.n_hits, 2)
        x8_mapped = mapper_pre(x8)
        assert x8_mapped is not None
        self.assertEqual(x8_mapped.double_num_squared, 128)
        self.assertEqual(square_hit_tracker.n_hits, 2)

        mapper_pre.reset_cache()
        x8_mapped = mapper_pre(x8)
        assert x8_mapped is not None
        self.assertEqual(x8_mapped.double_num_squared, 128)
        self.assertEqual(square_hit_tracker.n_hits, 3)

    def test_mapper_decorator_no_parens(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing parentheses"):

            @lambda_mapper
            def square(x: DataPoint) -> DataPoint:
                x.num_squared = x.num ** 2
                return x

    def test_mapper_with_args_kwargs(self) -> None:
        with self.assertRaises(ValueError):
            MapperWithArgs("my_mapper")

        with self.assertRaises(ValueError):
            MapperWithKwargs("my_mapper")


class TestGetHashable(unittest.TestCase):
    def test_get_hashable_hashable(self) -> None:
        x = (8, "abc")
        x_hashable = get_hashable(x)
        self.assertEqual(x, x_hashable)

    def test_get_hashable_dict(self) -> None:
        d = dict(a=8, b=dict(c=9, d="foo"))
        d_hashable = get_hashable(d)
        d_sub_expected = frozenset((("c", 9), ("d", "foo")))
        d_expected = frozenset((("a", 8), ("b", d_sub_expected)))
        self.assertEqual(d_hashable, d_expected)
        self.assertEqual(hash(d_hashable), hash(d_expected))

    def test_get_hashable_list(self) -> None:
        c = [8, dict(c=9, d="foo")]
        c_hashable = get_hashable(c)
        c_expected = (8, frozenset((("c", 9), ("d", "foo"))))
        self.assertEqual(c_hashable, c_expected)
        self.assertEqual(hash(c_hashable), hash(c_expected))

    def test_get_hashable_series(self) -> None:
        s = pd.Series(dict(a=8, b=dict(c=9, d="foo")), name="bar")
        s_hashable = get_hashable(s)
        s_sub_expected = frozenset((("c", 9), ("d", "foo")))
        s_expected = frozenset((("a", 8), ("b", s_sub_expected)))
        self.assertEqual(s_hashable, s_expected)
        self.assertEqual(hash(s_hashable), hash(s_expected))

    def test_get_hashable_series_with_doc(self) -> None:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Foo went to the bar.")
        s = pd.Series(dict(a=8, b=doc), name="baz")
        s_hashable = get_hashable(s)
        s_expected = frozenset((("a", 8), ("b", doc)))
        self.assertEqual(s_hashable, s_expected)
        self.assertEqual(hash(s_hashable), hash(s_expected))

    def test_get_hashable_ndarray(self) -> None:
        v = np.array([[3, 6, 9], [0.4, 0.8, 0.12]])
        x = (8, dict(a=v))
        x_hashable = get_hashable(x)
        x_expected = (8, frozenset((("a", v.data.tobytes()),)))
        self.assertEqual(x_hashable, x_expected)

    def test_get_hashable_unhashable(self) -> None:
        v = pd.DataFrame(dict(a=[4, 5], b=[1, 2]))
        x = (8, dict(a=v))
        with self.assertRaises(ValueError):
            get_hashable(x)
