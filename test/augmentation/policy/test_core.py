import unittest

from snorkel.augmentation.policy import RandomAugmentationPolicy


class TestAugmentationPolicy(unittest.TestCase):
    def test_random_augmentation_policy(self):
        policy = RandomAugmentationPolicy(2, sequence_length=2)
        n_samples = 100
        samples = [policy.generate() for _ in range(n_samples)]
        a_ct = samples.count([0, 0])
        b_ct = samples.count([0, 1])
        c_ct = samples.count([1, 0])
        d_ct = samples.count([1, 1])
        self.assertGreater(a_ct, 0)
        self.assertGreater(b_ct, 0)
        self.assertGreater(c_ct, 0)
        self.assertGreater(d_ct, 0)
        self.assertEqual(a_ct + b_ct + c_ct + d_ct, n_samples)
