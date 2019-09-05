import unittest

import numpy as np
import pandas as pd

from snorkel.labeling import LabelingFunction, LFAnalysis

L = [
    [-1, -1, 0, -1, -1, 0],
    [-1, -1, -1, 2, -1, -1],
    [2, -1, -1, -1, -1, 0],
    [1, -1, 2, -1, 0, 0],
    [-1, -1, -1, -1, -1, -1],
    [1, -1, 0, 2, 1, 0],
]

Y = [0, 1, 2, 0, 1, 2]

L_wo_abstain = [
    [3, 3, 3, 5, 4, 3],
    [3, 4, 5, 5, 4, 3],
    [5, 3, 4, 4, 5, 3],
    [4, 4, 5, 4, 3, 3],
    [3, 4, 3, 5, 4, 3],
    [5, 3, 3, 4, 4, 3],
]


def f(x):
    return -1


class TestAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.lfa = LFAnalysis(np.array(L))
        self.lfa_wo_abstain = LFAnalysis(np.array(L_wo_abstain))
        self.Y = np.array(Y)

    def test_label_coverage(self) -> None:
        self.assertEqual(self.lfa.label_coverage(), 5 / 6)

    def test_label_overlap(self) -> None:
        self.assertEqual(self.lfa.label_overlap(), 4 / 6)

    def test_label_conflict(self) -> None:
        self.assertEqual(self.lfa.label_conflict(), 3 / 6)

    def test_lf_polarities(self) -> None:
        polarities = self.lfa.lf_polarities()
        self.assertEqual(polarities, [[1, 2], [], [0, 2], [2], [0, 1], [0]])

    def test_lf_coverages(self) -> None:
        coverages = self.lfa.lf_coverages()
        coverages_expected = [3 / 6, 0, 3 / 6, 2 / 6, 2 / 6, 4 / 6]
        np.testing.assert_array_almost_equal(coverages, np.array(coverages_expected))

    def test_lf_overlaps(self) -> None:
        overlaps = self.lfa.lf_overlaps(normalize_by_coverage=False)
        overlaps_expected = [3 / 6, 0, 3 / 6, 1 / 6, 2 / 6, 4 / 6]
        np.testing.assert_array_almost_equal(overlaps, np.array(overlaps_expected))

        overlaps = self.lfa.lf_overlaps(normalize_by_coverage=True)
        overlaps_expected = [1, 0, 1, 1 / 2, 1, 1]
        np.testing.assert_array_almost_equal(overlaps, np.array(overlaps_expected))

    def test_lf_conflicts(self) -> None:
        conflicts = self.lfa.lf_conflicts(normalize_by_overlaps=False)
        conflicts_expected = [3 / 6, 0, 2 / 6, 1 / 6, 2 / 6, 3 / 6]
        np.testing.assert_array_almost_equal(conflicts, np.array(conflicts_expected))

        conflicts = self.lfa.lf_conflicts(normalize_by_overlaps=True)
        conflicts_expected = [1, 0, 2 / 3, 1, 1, 3 / 4]
        np.testing.assert_array_almost_equal(conflicts, np.array(conflicts_expected))

    def test_lf_empirical_accuracies(self) -> None:
        accs = self.lfa.lf_empirical_accuracies(self.Y)
        accs_expected = [1 / 3, 0, 1 / 3, 1 / 2, 1 / 2, 2 / 4]
        np.testing.assert_array_almost_equal(accs, np.array(accs_expected))

    def test_lf_empirical_probs(self) -> None:
        P_emp = self.lfa.lf_empirical_probs(self.Y, 3)
        P = np.array(
            [
                [[1 / 2, 1, 0], [0, 0, 0], [1 / 2, 0, 1 / 2], [0, 0, 1 / 2]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 1, 1 / 2], [1 / 2, 0, 1 / 2], [0, 0, 0], [1 / 2, 0, 0]],
                [[1, 1 / 2, 1 / 2], [0, 0, 0], [0, 0, 0], [0, 1 / 2, 1 / 2]],
                [[1 / 2, 1, 1 / 2], [1 / 2, 0, 0], [0, 0, 1 / 2], [0, 0, 0]],
                [[0, 1, 0], [1, 0, 1], [0, 0, 0], [0, 0, 0]],
            ]
        )
        np.testing.assert_array_almost_equal(P, P_emp)

    def test_lf_summary(self) -> None:
        df = self.lfa.lf_summary(self.Y, est_weights=None)
        df_expected = pd.DataFrame(
            {
                "Polarity": [[1, 2], [], [0, 2], [2], [0, 1], [0]],
                "Coverage": [3 / 6, 0, 3 / 6, 2 / 6, 2 / 6, 4 / 6],
                "Overlaps": [3 / 6, 0, 3 / 6, 1 / 6, 2 / 6, 4 / 6],
                "Conflicts": [3 / 6, 0, 2 / 6, 1 / 6, 2 / 6, 3 / 6],
                "Correct": [1, 0, 1, 1, 1, 2],
                "Incorrect": [2, 0, 2, 1, 1, 2],
                "Emp. Acc.": [1 / 3, 0, 1 / 3, 1 / 2, 1 / 2, 2 / 4],
            }
        )
        pd.testing.assert_frame_equal(df.round(6), df_expected.round(6))

        df = self.lfa.lf_summary(Y=None, est_weights=None)
        df_expected = pd.DataFrame(
            {
                "Polarity": [[1, 2], [], [0, 2], [2], [0, 1], [0]],
                "Coverage": [3 / 6, 0, 3 / 6, 2 / 6, 2 / 6, 4 / 6],
                "Overlaps": [3 / 6, 0, 3 / 6, 1 / 6, 2 / 6, 4 / 6],
                "Conflicts": [3 / 6, 0, 2 / 6, 1 / 6, 2 / 6, 3 / 6],
            }
        )
        pd.testing.assert_frame_equal(df.round(6), df_expected.round(6))

        est_weights = [1, 0, 1, 1, 1, 0.5]
        names = list("abcdef")
        lfs = [LabelingFunction(s, f) for s in names]
        lfa = LFAnalysis(np.array(L), lfs)
        df = lfa.lf_summary(self.Y, est_weights=est_weights)
        df_expected = pd.DataFrame(
            {
                "j": [0, 1, 2, 3, 4, 5],
                "Polarity": [[1, 2], [], [0, 2], [2], [0, 1], [0]],
                "Coverage": [3 / 6, 0, 3 / 6, 2 / 6, 2 / 6, 4 / 6],
                "Overlaps": [3 / 6, 0, 3 / 6, 1 / 6, 2 / 6, 4 / 6],
                "Conflicts": [3 / 6, 0, 2 / 6, 1 / 6, 2 / 6, 3 / 6],
                "Correct": [1, 0, 1, 1, 1, 2],
                "Incorrect": [2, 0, 2, 1, 1, 2],
                "Emp. Acc.": [1 / 3, 0, 1 / 3, 1 / 2, 1 / 2, 2 / 4],
                "Learned Weight": [1, 0, 1, 1, 1, 0.5],
            }
        ).set_index(pd.Index(names))
        pd.testing.assert_frame_equal(df.round(6), df_expected.round(6))

    def test_wrong_number_of_lfs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Number of LFs"):
            LFAnalysis(np.array(L), [LabelingFunction(s, f) for s in "ab"])

    def test_lf_summary_without_abstain(self) -> None:
        df = self.lfa_wo_abstain.lf_summary(self.Y + 4, est_weights=None)
        df_expected = pd.DataFrame(
            {
                "Polarity": [[3, 4, 5], [3, 4], [3, 4, 5], [4, 5], [3, 4, 5], [3]],
                "Coverage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "Overlaps": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "Conflicts": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "Correct": [1, 1, 1, 3, 1, 0],
                "Incorrect": [5, 5, 5, 3, 5, 6],
                "Emp. Acc.": [1 / 6, 1 / 6, 1 / 6, 3 / 6, 1 / 6, 0],
            }
        )
        pd.testing.assert_frame_equal(df.round(6), df_expected.round(6))
