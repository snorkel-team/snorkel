import unittest

import numpy as np

from snorkel.augmentation.apply import PandasTFApplier
from snorkel.augmentation.policy import ApplyOnePolicy
from snorkel.labeling.apply import PandasLFApplier
from snorkel.synthetic.synthetic_data import (
    generate_mog_dataset,
    generate_resampling_tfs,
    generate_single_feature_lfs,
)


class TestGenerateMOGDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 1000
        self.d = 20
        self.n_noise_dim = 2
        self.data = generate_mog_dataset(self.n, self.d, n_noise_dim=self.n_noise_dim)

    def test_output_types(self) -> None:
        # Test x
        x = self.data.loc[0].x
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape[0], self.d + self.n_noise_dim)

        # Test y
        self.assertIsInstance(self.data.loc[0].y, np.integer)

    def test_dataset_means(self) -> None:
        # Compute the means of the two gaussians and confirm that they
        # are different
        X1 = np.array(self.data[self.data.y == 1].x.tolist())
        X2 = np.array(self.data[self.data.y == 2].x.tolist())
        X1_mog = X1[:, : self.d]
        X2_mog = X2[:, : self.d]
        X1_noise = X1[:, self.d :]
        X2_noise = X2[:, self.d :]
        d_mog = np.linalg.norm(X1_mog.mean(axis=0) - X2_mog.mean(axis=0))
        d_noise = np.linalg.norm(X1_noise.mean(axis=0) - X2_noise.mean(axis=0))
        self.assertGreater(d_mog, 1)
        self.assertLess(d_noise, 1)


class TestGenerateSingleFeatureLFs(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 1000
        self.d = 20

    def test_perfect_lf(self) -> None:
        # Generate a dataset where the first feature is "perfect"
        cov = np.diag(np.random.random(self.d))
        cov[0, 0] = 0.0
        data = generate_mog_dataset(self.n, self.d, cov=cov)

        # Generate a single-feature LF and apply to the data
        lfs = generate_single_feature_lfs(1)
        lf_applier = PandasLFApplier(lfs)
        L = lf_applier.apply(data)

        # Get accuracies
        # TODO: Replace this with a generic utility function
        Y = data.y.values.astype(int)
        acc = np.where(L[:, 0].toarray().reshape(-1) == Y, 1, 0).sum() / self.n
        self.assertEqual(acc, 1.0)

    def test_abstain_rate(self) -> None:
        # Generate a dataset
        data = generate_mog_dataset(self.n, self.d)

        # Generate LFs with a specific abstain rate
        m = 10
        abstain_rate = 0.65
        lfs = generate_single_feature_lfs(m, abstain_rate=abstain_rate)
        lf_applier = PandasLFApplier(lfs)
        L = lf_applier.apply(data)

        # Get average abstain rate
        abstain_rate_est = 1 - np.where(L.todense() != 0, 1, 0).sum() / (self.n * m)
        self.assertAlmostEqual(abstain_rate, abstain_rate_est, delta=0.025)


class TestGenerateResamplingTFs(unittest.TestCase):
    def test_generate_resampling_tfs(self):
        data = generate_mog_dataset(10, 4)
        tf_dim = 0
        tfs = generate_resampling_tfs(dims=[tf_dim])
        policy = ApplyOnePolicy()
        applier = PandasTFApplier(tfs, policy, keep_original=False)
        data_augmented = applier.apply(data)
        X = np.array(data.x.tolist())
        X_augmented = np.array(data_augmented.x.tolist())
        col_is_same = np.all(np.isclose(X - X_augmented, 0), axis=0)
        for i, v in enumerate(col_is_same):
            if i == tf_dim:
                self.assertFalse(v)
            else:
                self.assertTrue(v)


if __name__ == "__main__":
    unittest.main()
