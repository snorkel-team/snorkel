import unittest
from types import SimpleNamespace
from typing import List

from snorkel.augmentation.policy import RandomAugmentationPolicy
from snorkel.augmentation.tf import TransformationFunctionMode, transformation_function
from snorkel.types import FieldMap


@transformation_function
def swap_a(words: List[str]) -> FieldMap:
    swaps = {"a": "b", "c": "d"}
    words_swapped = [swaps.get(w, w) for w in words]
    return dict(words_swapped=words_swapped)


@transformation_function
def swap_b(words: List[str]) -> FieldMap:
    swaps = {"b": "a", "d": "c"}
    words_swapped = [swaps.get(w, w) for w in words]
    return dict(words_swapped=words_swapped)


@transformation_function
def square(num: int) -> FieldMap:
    return dict(num=num ** 2)


class TestAugmentationPolicy(unittest.TestCase):
    def test_random_augmentation_policy(self):
        x = SimpleNamespace(words=["a", "b", "c", "d"])
        policy = RandomAugmentationPolicy([swap_a, swap_b])
        policy.set_tf_mode(TransformationFunctionMode.NAMESPACE)
        n_transformed = 50
        x_transformed = [policy.apply(x).words_swapped for _ in range(n_transformed)]
        a_ct = x_transformed.count(["a", "a", "c", "c"])
        b_ct = x_transformed.count(["b", "b", "d", "d"])
        self.assertGreater(a_ct, 0)
        self.assertGreater(b_ct, 0)
        self.assertEqual(a_ct + b_ct, n_transformed)

    def test_random_augmentation_policy_sequence(self):
        x = SimpleNamespace(num=2)
        policy = RandomAugmentationPolicy([square], sequence_length=2)
        policy.set_tf_mode(TransformationFunctionMode.NAMESPACE)
        x_transformed = policy.apply(x)
        self.assertEqual(x_transformed.num, 16)
