import unittest

import numpy as np

from snorkel.analysis import get_label_buckets, get_label_instances


class ErrorAnalysisTest(unittest.TestCase):
    def test_get_label_buckets(self) -> None:
        y1 = np.array([[2], [1], [3], [1], [1], [3]])
        y2 = np.array([1, 2, 3, 1, 2, 3])
        buckets = get_label_buckets(y1, y2)
        expected_buckets = {(2, 1): [0], (1, 2): [1, 4], (3, 3): [2, 5], (1, 1): [3]}
        expected_buckets = {k: np.array(v) for k, v in expected_buckets.items()}
        np.testing.assert_equal(buckets, expected_buckets)

        y1_1d = np.array([2, 1, 3, 1, 1, 3])
        buckets = get_label_buckets(y1_1d, y2)
        np.testing.assert_equal(buckets, expected_buckets)

    def test_get_label_buckets_multi(self) -> None:
        y1 = np.array([[2], [1], [3], [1], [1], [3]])
        y2 = np.array([1, 2, 3, 1, 2, 3])
        y3 = np.array([[3], [2], [1], [1], [2], [3]])
        buckets = get_label_buckets(y1, y2, y3)
        expected_buckets = {
            (2, 1, 3): [0],
            (1, 2, 2): [1, 4],
            (3, 3, 1): [2],
            (1, 1, 1): [3],
            (3, 3, 3): [5],
        }
        expected_buckets = {k: np.array(v) for k, v in expected_buckets.items()}
        np.testing.assert_equal(buckets, expected_buckets)

    def test_get_label_buckets_bad_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number of elements"):
            get_label_buckets(np.array([0, 1, 1]), np.array([1, 1]))

    def test_get_label_instances(self) -> None:
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y1 = np.array([1, 0, 0, 0])
        y2 = np.array([1, 1, 1, 0])
        instances = get_label_instances((0, 1), x, y1, y2)
        expected_instances = np.array([[3, 4], [5, 6]])
        np.testing.assert_equal(instances, expected_instances)

        x = np.array(["this", "is", "a", "test", "of", "multi"])
        y1 = np.array([[2], [1], [3], [1], [1], [3]])
        y2 = np.array([1, 2, 3, 1, 2, 3])
        y3 = np.array([[3], [2], [1], [1], [2], [3]])
        instances = get_label_instances((3, 3, 3), x, y1, y2, y3)
        expected_instances = np.array(["multi"])
        np.testing.assert_equal(instances, expected_instances)

    def test_get_label_instances_exceptions(self) -> None:
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y1 = np.array([1, 0, 0, 0])
        y2 = np.array([1, 1, 1, 0])
        instances = get_label_instances((2, 0), x, y1, y2)
        expected_instances = np.array([])
        np.testing.assert_equal(instances, expected_instances)

        with self.assertRaisesRegex(
            ValueError, "Number of lists must match the amount of labels in bucket"
        ):
            get_label_instances((1, 0), x, y1)

        x = np.array([[1, 2], [3, 4], [5, 6]])
        with self.assertRaisesRegex(
            ValueError,
            "Number of rows in x does not match number of elements in at least one label list",
        ):
            get_label_instances((1, 0), x, y1, y2)


if __name__ == "__main__":
    unittest.main()
