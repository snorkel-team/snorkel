import unittest

import numpy as np

from snorkel.labeling.apply import LFApplier
from snorkel.synthetic.synthetic_data import (
    generate_mog_dataset,
    generate_single_feature_LFs,
)


class TestGenerateMOGDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 1000
        self.d = 20
        self.data = generate_mog_dataset(self.n, self.d)

    def test_output_types(self) -> None:
        # Test x
        x = self.data[0].x
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape[0], self.d)

        # Test y
        self.assertIsInstance(self.data[0].y, np.integer)

    def test_dataset_means(self) -> None:
        # Compute the means of the two gaussians and confirm that they
        # are different
        m1 = np.mean([d.x for d in self.data if d.y == 1])
        m2 = np.mean([d.x for d in self.data if d.y == 2])
        dist = np.linalg.norm(m1 - m2)
        self.assertGreater(dist, 1)


class TestGenerateSingleFeatureLFs(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 1000
        self.d = 20

    def test_perfect_LF(self) -> None:
        # Generate a dataset where the first feature is "perfect"
        cov = np.diag(np.random.random(self.d))
        cov[0, 0] = 0.0
        data = generate_mog_dataset(self.n, self.d, cov=cov)

        # Generate a single-feature LF and apply to the data
        LFs = generate_single_feature_LFs(1)
        lf_applier = LFApplier(LFs)
        L = lf_applier.apply(data)

        # Get accuracies
        # TODO: Replace this with a generic utility function
        Y = np.array([d.y for d in data])
        acc = np.where(L[:, 0].toarray().reshape(-1) == Y, 1, 0).sum() / self.n
        self.assertEqual(acc, 1.0)

    def test_abstain_rate(self) -> None:
        # Generate a dataset
        data = generate_mog_dataset(self.n, self.d)

        # Generate LFs with a specific abstain rate
        m = 10
        abstain_rate = 0.65
        LFs = generate_single_feature_LFs(m, abstain_rate=abstain_rate)
        lf_applier = LFApplier(LFs)
        L = lf_applier.apply(data)

        # Get average abstain rate
        abstain_rate_est = 1 - np.where(L.todense() != 0, 1, 0).sum() / (self.n * m)
        self.assertAlmostEqual(abstain_rate, abstain_rate_est, delta=0.025)


if __name__ == "__main__":
    unittest.main()
