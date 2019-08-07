import unittest

from snorkel.augmentation import MeanFieldPolicy, RandomPolicy


class TestSamplingPolicy(unittest.TestCase):
    def test_random_policy(self):
        policy = RandomPolicy(2, sequence_length=2)
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

    def test_mean_field_policy(self):
        policy = MeanFieldPolicy(2, sequence_length=2, p=[1, 0])
        n_samples = 100
        samples = [policy.generate() for _ in range(n_samples)]
        self.assertEqual(samples.count([0, 0]), n_samples)
