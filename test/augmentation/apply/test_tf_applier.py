import unittest
from types import SimpleNamespace
from typing import List

import pandas as pd

from snorkel.augmentation import (
    ApplyOnePolicy,
    PandasTFApplier,
    RandomPolicy,
    TFApplier,
    transformation_function,
)
from snorkel.types import DataPoint


@transformation_function()
def square(x: DataPoint) -> DataPoint:
    x.num = x.num ** 2
    return x


@transformation_function()
def square_returns_none(x: DataPoint) -> DataPoint:
    if x.num == 2:
        return None
    x.num = x.num ** 2
    return x


@transformation_function()
def modify_in_place(x: DataPoint) -> DataPoint:
    x.d["my_key"] = 0
    return x


DATA = [1, 2, 3]
STR_DATA = ["x", "y", "z"]
DATA_IN_PLACE_EXPECTED = [(1 + i // 3) if i % 3 == 0 else 0 for i in range(9)]


def make_df(values: list, index: list, key: str = "num") -> pd.DataFrame:
    return pd.DataFrame({key: values}, index=index)


# NB: reconstruct each time to avoid inplace updates
def get_data_dict(data: List[int] = DATA):
    return [dict(my_key=num) for num in data]


class TestTFApplier(unittest.TestCase):
    def _get_x_namespace(self, data: List[int] = DATA) -> List[SimpleNamespace]:
        return [SimpleNamespace(num=num) for num in data]

    def _get_x_namespace_dict(self, data: List[int] = DATA) -> List[SimpleNamespace]:
        return [SimpleNamespace(d=d) for d in get_data_dict(data)]

    def test_tf_applier(self) -> None:
        data = self._get_x_namespace()
        policy = RandomPolicy(
            1, sequence_length=2, n_per_original=1, keep_original=False
        )
        applier = TFApplier([square], policy)
        data_augmented = applier.apply(data, progress_bar=False)
        self.assertEqual(data_augmented, self._get_x_namespace([1, 16, 81]))
        self.assertEqual(data, self._get_x_namespace())

        data_augmented = applier.apply(data, progress_bar=True)
        self.assertEqual(data_augmented, self._get_x_namespace([1, 16, 81]))
        self.assertEqual(data, self._get_x_namespace())

    def test_tf_applier_keep_original(self) -> None:
        data = self._get_x_namespace()
        policy = RandomPolicy(
            1, sequence_length=2, n_per_original=2, keep_original=True
        )
        applier = TFApplier([square], policy)
        data_augmented = applier.apply(data, progress_bar=False)
        vals = [1, 1, 1, 2, 16, 16, 3, 81, 81]
        self.assertEqual(data_augmented, self._get_x_namespace(vals))
        self.assertEqual(data, self._get_x_namespace())

    def test_tf_applier_returns_none(self) -> None:
        data = self._get_x_namespace()
        policy = RandomPolicy(
            1, sequence_length=2, n_per_original=2, keep_original=True
        )
        applier = TFApplier([square_returns_none], policy)
        data_augmented = applier.apply(data, progress_bar=False)
        vals = [1, 1, 1, 2, 3, 81, 81]
        self.assertEqual(data_augmented, self._get_x_namespace(vals))
        self.assertEqual(data, self._get_x_namespace())

    def test_tf_applier_keep_original_modify_in_place(self) -> None:
        data = self._get_x_namespace_dict()
        policy = ApplyOnePolicy(n_per_original=2, keep_original=True)
        applier = TFApplier([modify_in_place], policy)
        data_augmented = applier.apply(data, progress_bar=False)
        self.assertEqual(
            data_augmented, self._get_x_namespace_dict(DATA_IN_PLACE_EXPECTED)
        )
        self.assertEqual(data, self._get_x_namespace_dict())

    def test_tf_applier_generator(self) -> None:
        data = self._get_x_namespace()
        policy = RandomPolicy(
            1, sequence_length=2, n_per_original=2, keep_original=False
        )
        applier = TFApplier([square], policy)
        batches_expected = [[1, 1, 16, 16], [81, 81]]
        gen = applier.apply_generator(data, batch_size=2)
        for batch, batch_expected in zip(gen, batches_expected):
            self.assertEqual(batch, self._get_x_namespace(batch_expected))
        self.assertEqual(data, self._get_x_namespace())

    def test_tf_applier_keep_original_generator(self) -> None:
        data = self._get_x_namespace()
        policy = RandomPolicy(
            1, sequence_length=2, n_per_original=2, keep_original=True
        )
        applier = TFApplier([square], policy)
        batches_expected = [[1, 1, 1, 2, 16, 16], [3, 81, 81]]
        gen = applier.apply_generator(data, batch_size=2)
        for batch, batch_expected in zip(gen, batches_expected):
            self.assertEqual(batch, self._get_x_namespace(batch_expected))
        self.assertEqual(data, self._get_x_namespace())

    def test_tf_applier_returns_none_generator(self) -> None:
        data = self._get_x_namespace()
        policy = RandomPolicy(
            1, sequence_length=2, n_per_original=2, keep_original=True
        )
        applier = TFApplier([square_returns_none], policy)
        batches_expected = [[1, 1, 1, 2], [3, 81, 81]]
        gen = applier.apply_generator(data, batch_size=2)
        for batch, batch_expected in zip(gen, batches_expected):
            self.assertEqual(batch, self._get_x_namespace(batch_expected))
        self.assertEqual(data, self._get_x_namespace())

    def test_tf_applier_keep_original_modify_in_place_generator(self) -> None:
        data = self._get_x_namespace_dict()
        policy = ApplyOnePolicy(n_per_original=2, keep_original=True)
        applier = TFApplier([modify_in_place], policy)
        batches_expected = [DATA_IN_PLACE_EXPECTED[:6], DATA_IN_PLACE_EXPECTED[6:]]
        gen = applier.apply_generator(data, batch_size=2)
        for batch, batch_expected in zip(gen, batches_expected):
            self.assertEqual(batch, self._get_x_namespace_dict(batch_expected))
        self.assertEqual(data, self._get_x_namespace_dict())


class TestPandasTFApplier(unittest.TestCase):
    def _get_x_df(self):
        return pd.DataFrame(dict(num=DATA))

    def _get_x_df_with_str(self):
        return pd.DataFrame(dict(num=DATA, strs=STR_DATA))

    def _get_x_df_dict(self):
        return pd.DataFrame(dict(d=get_data_dict()))

    def test_tf_applier_pandas(self):
        df = self._get_x_df_with_str()
        policy = RandomPolicy(
            1, sequence_length=2, n_per_original=1, keep_original=False
        )
        applier = PandasTFApplier([square], policy)
        df_augmented = applier.apply(df, progress_bar=False)
        df_expected = pd.DataFrame(
            dict(num=[1, 16, 81], strs=STR_DATA), index=[0, 1, 2]
        )
        self.assertEqual(df_augmented.num.dtype, "int64")
        pd.testing.assert_frame_equal(df_augmented, df_expected)
        pd.testing.assert_frame_equal(df, self._get_x_df_with_str())

        df_augmented = applier.apply(df, progress_bar=True)
        df_expected = pd.DataFrame(
            dict(num=[1, 16, 81], strs=STR_DATA), index=[0, 1, 2]
        )
        pd.testing.assert_frame_equal(df_augmented, df_expected)
        pd.testing.assert_frame_equal(df, self._get_x_df_with_str())

    def test_tf_applier_pandas_keep_original(self):
        df = self._get_x_df()
        policy = RandomPolicy(
            1, sequence_length=2, n_per_original=2, keep_original=True
        )
        applier = PandasTFApplier([square], policy)
        df_augmented = applier.apply(df, progress_bar=False)
        df_expected = pd.DataFrame(
            dict(num=[1, 1, 1, 2, 16, 16, 3, 81, 81]), index=[0, 0, 0, 1, 1, 1, 2, 2, 2]
        )
        self.assertEqual(df_augmented.num.dtype, "int64")
        pd.testing.assert_frame_equal(df_augmented, df_expected)
        pd.testing.assert_frame_equal(df, self._get_x_df())

    def test_tf_applier_returns_none(self):
        df = self._get_x_df()
        policy = RandomPolicy(
            1, sequence_length=2, n_per_original=2, keep_original=True
        )
        applier = PandasTFApplier([square_returns_none], policy)
        df_augmented = applier.apply(df, progress_bar=False)
        df_expected = pd.DataFrame(
            dict(num=[1, 1, 1, 2, 3, 81, 81]), index=[0, 0, 0, 1, 2, 2, 2]
        )
        self.assertEqual(df_augmented.num.dtype, "int64")
        pd.testing.assert_frame_equal(df_augmented, df_expected)
        pd.testing.assert_frame_equal(df, self._get_x_df())

    def test_tf_applier_pandas_modify_in_place(self):
        df = self._get_x_df_dict()
        policy = ApplyOnePolicy(n_per_original=2, keep_original=True)
        applier = PandasTFApplier([modify_in_place], policy)
        df_augmented = applier.apply(df, progress_bar=False)
        idx = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        df_expected = pd.DataFrame(
            dict(d=get_data_dict(DATA_IN_PLACE_EXPECTED)), index=idx
        )
        pd.testing.assert_frame_equal(df_augmented, df_expected)
        pd.testing.assert_frame_equal(df, self._get_x_df_dict())

    def test_tf_applier_pandas_generator(self):
        df = self._get_x_df_with_str()
        policy = RandomPolicy(
            1, sequence_length=2, n_per_original=2, keep_original=False
        )
        applier = PandasTFApplier([square], policy)
        gen = applier.apply_generator(df, batch_size=2)
        df_expected = [
            pd.DataFrame(
                {"num": [1, 1, 16, 16], "strs": ["x", "x", "y", "y"]},
                index=[0, 0, 1, 1],
            ),
            pd.DataFrame({"num": [81, 81], "strs": ["z", "z"]}, index=[2, 2]),
        ]
        for df_batch, df_batch_expected in zip(gen, df_expected):
            self.assertEqual(df_batch.num.dtype, "int64")
            pd.testing.assert_frame_equal(df_batch, df_batch_expected)
        pd.testing.assert_frame_equal(df, self._get_x_df_with_str())

    def test_tf_applier_pandas_keep_original_generator(self):
        df = self._get_x_df()
        policy = RandomPolicy(
            1, sequence_length=2, n_per_original=2, keep_original=True
        )
        applier = PandasTFApplier([square], policy)
        gen = applier.apply_generator(df, batch_size=2)
        df_expected = [
            make_df([1, 1, 1, 2, 16, 16], [0, 0, 0, 1, 1, 1]),
            make_df([3, 81, 81], [2, 2, 2]),
        ]
        for df_batch, df_batch_expected in zip(gen, df_expected):
            pd.testing.assert_frame_equal(df_batch, df_batch_expected)
        pd.testing.assert_frame_equal(df, self._get_x_df())

    def test_tf_applier_returns_none_generator(self):
        df = self._get_x_df()
        policy = RandomPolicy(
            1, sequence_length=2, n_per_original=2, keep_original=True
        )
        applier = PandasTFApplier([square_returns_none], policy)
        gen = applier.apply_generator(df, batch_size=2)
        df_expected = [
            make_df([1, 1, 1, 2], [0, 0, 0, 1]),
            make_df([3, 81, 81], [2, 2, 2]),
        ]
        for df_batch, df_batch_expected in zip(gen, df_expected):
            pd.testing.assert_frame_equal(df_batch, df_batch_expected)
        pd.testing.assert_frame_equal(df, self._get_x_df())

    def test_tf_applier_pandas_modify_in_place_generator(self):
        df = self._get_x_df_dict()
        policy = ApplyOnePolicy(n_per_original=2, keep_original=True)
        applier = PandasTFApplier([modify_in_place], policy)
        gen = applier.apply_generator(df, batch_size=2)
        idx = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        df_expected = [
            make_df(get_data_dict(DATA_IN_PLACE_EXPECTED[:6]), idx[:6], key="d"),
            make_df(get_data_dict(DATA_IN_PLACE_EXPECTED[6:]), idx[6:], key="d"),
        ]
        for df_batch, df_batch_expected in zip(gen, df_expected):
            pd.testing.assert_frame_equal(df_batch, df_batch_expected)
        pd.testing.assert_frame_equal(df, self._get_x_df_dict())
