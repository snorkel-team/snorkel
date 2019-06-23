import unittest
from types import SimpleNamespace

import pandas as pd

from snorkel.augmentation.apply import PandasTFApplier, TFApplier
from snorkel.augmentation.policy import RandomAugmentationPolicy
from snorkel.augmentation.tf import transformation_function
from snorkel.types import DataPoint


@transformation_function
def square(x: DataPoint) -> DataPoint:
    x.num = x.num ** 2
    return x


@transformation_function
def square_returns_none(x: DataPoint) -> DataPoint:
    if x.num == 2:
        return None
    x.num = x.num ** 2
    return x


@transformation_function
def modify_in_place(x: DataPoint) -> DataPoint:
    x.d["my_key"] = 0
    return x


policy = RandomAugmentationPolicy(1, sequence_length=2)
policy_modify_in_place = RandomAugmentationPolicy(1, sequence_length=1)


DATA = [1, 2, 3]
DATA_IN_PLACE_EXPECTED = [
    dict(my_key=(1 + i // 3) if i % 3 == 0 else 0) for i in range(9)
]


# NB: reconstruct each time to avoid inplace updates
def get_data_dict():
    return [dict(my_key=num) for num in DATA]


class TestTFApplier(unittest.TestCase):
    def _get_x_namespace(self):
        return [SimpleNamespace(num=num) for num in DATA]

    def _get_x_namespace_dict(self):
        return [SimpleNamespace(d=d) for d in get_data_dict()]

    def test_tf_applier(self):
        data = self._get_x_namespace()
        applier = TFApplier([square], policy, k=1, keep_original=False)
        data_augmented = applier.apply(data)
        vals = [x.num for x in data_augmented]
        self.assertEqual(vals, [1, 16, 81])
        self.assertEqual([x.num for x in data], DATA)

    def test_tf_applier_multi(self):
        data = self._get_x_namespace()
        applier = TFApplier([square], policy, k=2, keep_original=False)
        data_augmented = applier.apply(data)
        vals = [x.num for x in data_augmented]
        self.assertEqual(vals, [1, 1, 16, 16, 81, 81])
        self.assertEqual([x.num for x in data], DATA)

    def test_tf_applier_keep_original(self):
        data = self._get_x_namespace()
        applier = TFApplier([square], policy, k=2, keep_original=True)
        data_augmented = applier.apply(data)
        vals = [x.num for x in data_augmented]
        self.assertEqual(vals, [1, 1, 1, 2, 16, 16, 3, 81, 81])
        self.assertEqual([x.num for x in data], DATA)

    def test_tf_applier_returns_none(self):
        data = self._get_x_namespace()
        applier = TFApplier([square_returns_none], policy, k=2, keep_original=True)
        data_augmented = applier.apply(data)
        vals = [x.num for x in data_augmented]
        self.assertEqual(vals, [1, 1, 1, 2, 3, 81, 81])
        self.assertEqual([x.num for x in data], DATA)

    def test_tf_applier_keep_original_modify_in_place(self):
        data = self._get_x_namespace_dict()
        applier = TFApplier(
            [modify_in_place], policy_modify_in_place, k=2, keep_original=True
        )
        data_augmented = applier.apply(data)
        self.assertTrue(
            all(x.d == d for x, d in zip(data_augmented, DATA_IN_PLACE_EXPECTED))
        )
        self.assertTrue(all(x.d == y for x, y in zip(data, get_data_dict())))


class TestPandasTFApplier(unittest.TestCase):
    def _get_x_df(self):
        return pd.DataFrame(dict(num=DATA))

    def _get_x_df_dict(self):
        return pd.DataFrame(dict(d=get_data_dict()))

    def test_tf_applier_pandas(self):
        df = self._get_x_df()
        applier = PandasTFApplier([square], policy, k=1, keep_original=False)
        df_augmented = applier.apply(df)
        df_expected = pd.DataFrame(dict(num=[1, 16, 81]), index=[0, 1, 2])
        self.assertTrue(df_augmented.equals(df_expected))
        self.assertEqual(df.num.tolist(), DATA)

    def test_tf_applier_pandas_multi(self):
        df = self._get_x_df()
        applier = PandasTFApplier([square], policy, k=2, keep_original=False)
        df_augmented = applier.apply(df)
        df_expected = pd.DataFrame(
            dict(num=[1, 1, 16, 16, 81, 81]), index=[0, 0, 1, 1, 2, 2]
        )
        self.assertTrue(df_augmented.equals(df_expected))
        self.assertEqual(df.num.tolist(), DATA)

    def test_tf_applier_pandas_keep_original(self):
        df = self._get_x_df()
        applier = PandasTFApplier([square], policy, k=2, keep_original=True)
        df_augmented = applier.apply(df)
        df_expected = pd.DataFrame(
            dict(num=[1, 1, 1, 2, 16, 16, 3, 81, 81]), index=[0, 0, 0, 1, 1, 1, 2, 2, 2]
        )
        self.assertTrue(df_augmented.equals(df_expected))
        self.assertEqual(df.num.tolist(), DATA)

    def test_tf_applier_returns_none(self):
        df = self._get_x_df()
        applier = PandasTFApplier(
            [square_returns_none], policy, k=2, keep_original=True
        )
        df_augmented = applier.apply(df)
        df_expected = pd.DataFrame(
            dict(num=[1, 1, 1, 2, 3, 81, 81]), index=[0, 0, 0, 1, 2, 2, 2]
        )
        self.assertTrue(df_augmented.equals(df_expected))
        self.assertEqual(df.num.tolist(), DATA)

    def test_tf_applier_pandas_modify_in_place(self):
        df = self._get_x_df_dict()
        applier = PandasTFApplier(
            [modify_in_place], policy_modify_in_place, k=2, keep_original=True
        )
        df_augmented = applier.apply(df)
        idx = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        df_expected = pd.DataFrame(dict(d=DATA_IN_PLACE_EXPECTED), index=idx)
        self.assertTrue(df_augmented.equals(df_expected))
        self.assertEqual(df.d.tolist(), get_data_dict())
