import unittest

import numpy as np
from sklearn.model_selection import GridSearchCV

from snorkel.labeling import SklearnLabelModel


class LabelModelWrapperTest(unittest.TestCase):
    def test_create_param_search_data(self) -> None:
        L_train = np.array([[1, 1, 0], [-1, -1, 1], [0, 0, 1], [1, 1, 0]])
        L_dev = np.array([[-1, 1, 1], [0, -1, 0]])
        Y_dev = np.array([0, 1])

        # combined L, Y
        L_true = np.array(
            [[1, 1, 0], [-1, -1, 1], [0, 0, 1], [1, 1, 0], [-1, 1, 1], [0, -1, 0]]
        )
        Y_true = np.array([-1, -1, -1, -1, 0, 1])

        label_model = SklearnLabelModel()
        L, Y, cv_split = label_model.create_param_search_data(L_train, L_dev, Y_dev)
        np.testing.assert_array_equal(L, L_true)
        np.testing.assert_array_equal(Y, Y_true)
        np.testing.assert_array_equal(cv_split.get_n_splits(), 1)

        L_dev = np.array([[-1, 1, 1], [0, -1, 0], [1, 1, 0]])
        with self.assertRaisesRegex(ValueError, "Num. datapoints in Y_dev and L_dev"):
            L, Y, cv_split = label_model.create_param_search_data(L_train, L_dev, Y_dev)

    def test_search(self):
        L_train = np.array(
            [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 1]]
        )
        L_dev = np.array([[1, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]])
        Y_dev = np.array([1, 1, 0, 1])

        label_model = SklearnLabelModel()
        L, Y, cv_split = label_model.create_param_search_data(L_train, L_dev, Y_dev)

        param_grid = [{"n_epochs": [5, 100], "lr": [1e-10, 1e-2], "metric": ["f1"]}]

        clf = GridSearchCV(label_model, param_grid, cv=cv_split)
        clf.fit(L, Y)

        self.assertEqual(
            clf.best_params_, {"lr": 0.01, "metric": "f1", "n_epochs": 100}
        )
        self.assertEqual(clf.best_index_, 3)
        self.assertEqual(clf.n_splits_, 1)


if __name__ == "__main__":
    unittest.main()
