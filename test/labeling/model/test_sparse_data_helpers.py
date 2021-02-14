import unittest
from typing import List, Tuple

import numpy as np

from snorkel.labeling.model import LabelModel
from snorkel.labeling.model.sparse_data_helpers import (
    KnownDimensions,
    SparseLabelModel,
)
from snorkel.labeling.model import sparse_data_helpers
from scipy.sparse import csr_matrix

class SparseHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        known_dimensions = KnownDimensions(
            num_classes=3, num_examples=5, num_functions=7
        )

        possible_function_values = np.eye(known_dimensions.num_classes)
        choice_set = np.random.choice(known_dimensions.num_classes,size=[known_dimensions.num_functions,known_dimensions.num_examples])
        cooccurence = np.hstack(
            possible_function_values[choice_set]
        )
        cls.known_dimensions = known_dimensions

        cls.L_ind = cooccurence
        cls.O_counts = (cls.L_ind.T @ cls.L_ind)
        cls.O = cls.O_counts /cls.known_dimensions.num_examples
        model = LabelModel()
        model._set_constants(known_dimensions=cls.known_dimensions)
        model._create_tree()
        model._generate_O_from_L_aug(cls.L_ind)
        cls.model_O = model.O.detach().numpy()

    def test_properly_constructed(self):
        self.assertEqual(
            (self.known_dimensions.num_examples, self.known_dimensions.num_events,), self.L_ind.shape
        )
    def test_that_tests_generate_O_correctly(self):
        np.testing.assert_almost_equal(self.model_O,self.O)
    def test_prepare_objective_from_sparse_event_cooccurence(self):
        sparse_model = SparseLabelModel()
        sparse_L_ind = csr_matrix(self.L_ind)
        data = sparse_L_ind.data
        indptr = sparse_L_ind.indptr
        indices = sparse_L_ind.indices
        tuples: List[Tuple[int, int, int]] = []
        for row in range(self.L_ind.shape[0]):
            col_range = indices[sparse_L_ind.indptr[row]:indptr[row+1]]
            val_range = data[indptr[row]:indptr[row+1]]
            self.assertEqual(len(col_range),len(val_range))
            for col,value in zip(col_range,val_range):
                self.assertEqual(value, self.L_ind[row, col])
                tuples.append((row,col, value))


        sparse_L_ind = sparse_model._prepare_sparse_L_ind(known_dimensions=self.known_dimensions,sparse_event_cooccurence=tuples)
        np.testing.assert_almost_equal(self.L_ind, sparse_L_ind.todense())
        calculated_obective = sparse_model._prepare_objective_from_sparse_event_cooccurence(tuples,known_dimensions=self.known_dimensions)
        self.assertEqual(self.model_O.shape,calculated_obective.shape)
        np.testing.assert_almost_equal(self.model_O, calculated_obective)
    def test_train_model_from_sparse_O(self):
        sparse_model = SparseLabelModel()
        sparse_model.train_model_from_known_objective(objective=self.O,
                                                                   known_dimensions=self.known_dimensions)

        mu_cpu = sparse_model.mu.detach().numpy()
        self.assertEqual((self.known_dimensions.num_events,self.known_dimensions.num_classes),mu_cpu.shape)
        with self.assertRaises(NotImplementedError):
            sparse_model.get_weights()
    def test_sparse_prediction(self):
        sparse_model = SparseLabelModel()
        sparse_model.train_model_from_known_objective(objective=self.O,
                                                                   known_dimensions=self.known_dimensions)
        events = np.arange(self.known_dimensions.num_events)
        np.random.shuffle(events) # for fun
        cliquesets = set(tuple(events[i:i**i % self.known_dimensions.num_events]) for i in range(self.known_dimensions.num_events))
        prediction_dict = sparse_model.predict_probs_from_cliqueset(cliqueset_indice_list=cliquesets)
        for cs in cliquesets:
            self.assertIn(tuple(cs),prediction_dict)
        self.assertEqual(len(cliquesets),len(prediction_dict.keys())) #if all elements match, and they have the same size, they are equal




