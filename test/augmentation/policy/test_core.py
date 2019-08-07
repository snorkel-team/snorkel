import unittest

from snorkel.augmentation import ApplyAllPolicy, ApplyEachPolicy


class TestPolicy(unittest.TestCase):
    def test_apply_each_policy(self):
        policy = ApplyEachPolicy(3, keep_original=True)
        samples = policy.generate_for_example()
        self.assertEqual(samples, [[], [0], [1], [2]])

        policy = ApplyEachPolicy(3, keep_original=False)
        samples = policy.generate_for_example()
        self.assertEqual(samples, [[0], [1], [2]])

    def test_apply_all_policy(self):
        policy = ApplyAllPolicy(3, n_per_original=2, keep_original=False)
        samples = policy.generate_for_example()
        self.assertEqual(samples, [[0, 1, 2], [0, 1, 2]])
