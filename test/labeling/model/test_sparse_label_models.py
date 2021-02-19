import unittest
from typing import List

import numpy as np
import pytest

from snorkel.labeling.model import LabelModel
from snorkel.labeling.model.sparse_label_model.base_sparse_label_model import (
    BaseSparseLabelModel,
)
from snorkel.labeling.model.sparse_label_model.sparse_event_pair_label_model import (
    SparseEventPairLabelModel,
)
from snorkel.labeling.model.sparse_label_model.sparse_example_eventlist_label_model import (
    SparseExampleEventListLabelModel,
)
from snorkel.labeling.model.sparse_label_model.sparse_label_model_helpers import (
    EventCooccurence,
    ExampleEventListOccurence,
    KnownDimensions,
    UnnormalizedObjectiveException,
)
from snorkel.synthetic.synthetic_data import generate_simple_label_matrix


class SparseModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        known_dimensions = KnownDimensions(
            num_classes=7, num_examples=1000, num_functions=10
        )

        possible_function_values = np.eye(known_dimensions.num_classes)
        choice_set = np.random.choice(
            known_dimensions.num_classes,
            size=[known_dimensions.num_functions, known_dimensions.num_examples],
        )
        cooccurence = np.hstack(possible_function_values[choice_set])
        cls.known_dimensions = known_dimensions

        cls.L_ind = cooccurence
        cls.O_counts = cls.L_ind.T @ cls.L_ind
        cls.O = cls.O_counts / cls.known_dimensions.num_examples
        model = LabelModel()
        model._set_constants(known_dimensions=cls.known_dimensions)
        model._create_tree()
        model._generate_O_from_L_aug(cls.L_ind)
        cls.model_O = model.O.detach().numpy()

    def test_properly_constructed(self):
        self.assertEqual(
            (self.known_dimensions.num_examples, self.known_dimensions.num_events,),
            self.L_ind.shape,
        )


class BaseSparseLabelModelTest(SparseModelTestCase):
    def test_fit_on_objective_raises_on_malformed(self):
        bad_objective = np.random.randint(
            -5,
            5,
            size=[self.known_dimensions.num_events, self.known_dimensions.num_events],
        )
        with self.assertRaises(UnnormalizedObjectiveException):
            sparse_model = BaseSparseLabelModel()
            sparse_model.fit_from_objective(
                objective=bad_objective, known_dimensions=self.known_dimensions
            )

    def test_that_tests_generate_O_correctly(self):
        np.testing.assert_almost_equal(self.model_O, self.O)

    def test_train_model_from_sparse_O(self):
        sparse_model = BaseSparseLabelModel()
        sparse_model.fit_from_objective(
            objective=self.O, known_dimensions=self.known_dimensions
        )

        mu_cpu = sparse_model.mu.detach().numpy()
        self.assertEqual(
            (self.known_dimensions.num_events, self.known_dimensions.num_classes),
            mu_cpu.shape,
        )
        with self.assertRaises(NotImplementedError):
            sparse_model.get_weights()

    def test_sparse_predict_proba(self):
        sparse_model = BaseSparseLabelModel()
        sparse_model.fit_from_objective(
            objective=self.O, known_dimensions=self.known_dimensions
        )
        events = np.arange(self.known_dimensions.num_events)
        np.random.shuffle(events)  # for fun
        cliquesets = set(
            tuple(events[i : i ** i % self.known_dimensions.num_events])
            for i in range(self.known_dimensions.num_events)
        )
        prediction_dict = sparse_model.predict_proba_from_cliqueset(
            cliquesets=cliquesets
        )
        self.assertIsInstance(prediction_dict, tuple)
        self.assertEqual(2, len(prediction_dict))
        cliqueset_list, probs_array = prediction_dict
        self.assertIsInstance(cliqueset_list, list)
        self.assertIsInstance(probs_array, np.ndarray)
        self.assertEqual(len(cliqueset_list), probs_array.shape[0])
        self.assertEqual(self.known_dimensions.num_classes, probs_array.shape[1])

    def test_sparse_predict_with_classes(self):
        sparse_model = BaseSparseLabelModel()
        sparse_model.fit_from_objective(
            objective=self.O, known_dimensions=self.known_dimensions
        )
        events = np.arange(self.known_dimensions.num_events)
        np.random.shuffle(events)  # for fun
        cliquesets = set(
            tuple(events[i : i ** i % self.known_dimensions.num_events])
            for i in range(self.known_dimensions.num_events)
        )
        cliqueset_list, probs, classes = sparse_model.predict(
            cliquesets, return_probs=True
        )
        self.assertEqual(len(cliquesets), len(cliqueset_list))
        self.assertEqual(len(cliquesets), (probs.shape[0]))
        self.assertEqual(len(cliquesets), (classes.shape[0]))
        self.assertIsInstance(classes, np.ndarray)
        self.assertEqual(self.known_dimensions.num_classes, probs.shape[1])


@pytest.mark.complex
class SparseEventPairLabelModelTest(SparseModelTestCase):
    def test_sparse_and_regular_make_same_objective(self):
        np.random.seed(123)
        P, Y, L = generate_simple_label_matrix(
            self.known_dimensions.num_examples,
            self.known_dimensions.num_functions,
            self.known_dimensions.num_classes,
        )
        sparse_event_occurence: List[EventCooccurence] = []
        label_model = LabelModel(cardinality=self.known_dimensions.num_classes)
        label_model._set_constants(L)
        L_shift = L + 1
        label_model_lind = label_model._create_L_ind(L_shift)
        co_oc_matrix = label_model_lind.T @ label_model_lind
        for a_id, cols in enumerate(co_oc_matrix):
            for b_id, freq in enumerate(cols):
                sparse_event_occurence.append(
                    EventCooccurence(a_id, b_id, frequency=freq)
                )

        sparse_model = SparseEventPairLabelModel()
        sparse_model._set_constants(known_dimensions=self.known_dimensions)

        sparse_model_objective = sparse_model._prepare_objective_from_sparse_event_cooccurence(
            known_dimensions=self.known_dimensions,
            sparse_event_occurence=sparse_event_occurence,
        )
        self.assertEqual(label_model.n, sparse_model.n)
        self.assertEqual(label_model.m, sparse_model.m)
        self.assertEqual(label_model.cardinality, sparse_model.cardinality)
        label_model._generate_O(L_shift,)
        label_model_O = label_model.O.detach().numpy()
        np.testing.assert_almost_equal(label_model_O, sparse_model_objective)

    def test_sparse_and_regular_make_same_probs(self) -> None:
        """Test the LabelModel's estimate of P and Y on a simple synthetic dataset."""
        np.random.seed(123)
        P, Y, L = generate_simple_label_matrix(
            self.known_dimensions.num_examples,
            self.known_dimensions.num_functions,
            self.known_dimensions.num_classes,
        )
        sparse_event_occurence: List[EventCooccurence] = []
        label_model = LabelModel(cardinality=self.known_dimensions.num_classes)
        label_model._set_constants(L)
        L_shift = L + 1
        label_model_lind = label_model._create_L_ind(L_shift)
        co_oc_matrix = label_model_lind.T @ label_model_lind
        for a_id, cols in enumerate(co_oc_matrix):
            for b_id, freq in enumerate(cols):
                sparse_event_occurence.append(
                    EventCooccurence(a_id, b_id, frequency=freq)
                )

        sparse_model = SparseEventPairLabelModel()

        sparse_model.fit_from_sparse_event_cooccurrence(
            sparse_event_occurence=sparse_event_occurence,
            known_dimensions=self.known_dimensions,
            n_epochs=200,
            lr=0.01,
            seed=123,
        )
        label_model = LabelModel(cardinality=self.known_dimensions.num_classes)
        label_model.fit(L, n_epochs=200, lr=0.01, seed=123)
        P_lm = label_model.get_conditional_probs()
        P_slm = sparse_model.get_conditional_probs()
        np.testing.assert_array_almost_equal(
            P_slm, P_lm,
        )


@pytest.mark.complex
class SparseExampleEventListLabelModelTest(SparseModelTestCase):
    def test_sparse_and_regular_make_same_l_ind_and_o(self):
        np.random.seed(123)
        P, Y, L = generate_simple_label_matrix(
            self.known_dimensions.num_examples,
            self.known_dimensions.num_functions,
            self.known_dimensions.num_classes,
        )
        example_event_lists: List[ExampleEventListOccurence] = []
        label_model = LabelModel(cardinality=self.known_dimensions.num_classes)
        label_model._set_constants(L)
        L_shift = L + 1
        label_model_lind = label_model._create_L_ind(L_shift)

        for example_num, example in enumerate(L):
            event_list = []
            for func_id, cls_id in enumerate(example):
                if (cls_id) > -1:
                    event_id = func_id * self.known_dimensions.num_classes + cls_id
                    event_list.append(event_id)
            example_event_lists.append((ExampleEventListOccurence(event_list)))

        sparse_model = SparseExampleEventListLabelModel()
        sparse_model._set_constants(known_dimensions=self.known_dimensions)
        sparse_model_lind = sparse_model.get_l_ind(
            known_dimensions=self.known_dimensions,
            example_events_list=example_event_lists,
            return_array=True,
        )
        sparse_model_objective = sparse_model._prepare_objective_from_sparse_example_eventlist(
            known_dimensions=self.known_dimensions,
            example_events_list=example_event_lists,
        )
        np.testing.assert_equal(label_model_lind, sparse_model_lind)
        np.testing.assert_equal(label_model_lind, sparse_model_lind)
        self.assertEqual(label_model.n, sparse_model.n)
        self.assertEqual(label_model.m, sparse_model.m)
        self.assertEqual(label_model.cardinality, sparse_model.cardinality)
        label_model._generate_O(L_shift,)
        label_model_O = label_model.O.detach().numpy()
        np.testing.assert_almost_equal(label_model_O, sparse_model_objective)

    def test_sparse_and_regular_make_same_probs(self) -> None:
        """Test the LabelModel's estimate of P and Y on a simple synthetic dataset."""
        np.random.seed(123)
        P, Y, L = generate_simple_label_matrix(
            self.known_dimensions.num_examples,
            self.known_dimensions.num_functions,
            self.known_dimensions.num_classes,
        )
        example_event_lists: List[ExampleEventListOccurence] = []

        for example_num, example in enumerate(L):
            event_list = []
            for func_id, cls_id in enumerate(example):
                if (cls_id) > -1:
                    event_id = func_id * self.known_dimensions.num_classes + cls_id
                    event_list.append(event_id)
            example_event_lists.append((ExampleEventListOccurence(event_list)))

        sparse_model = SparseExampleEventListLabelModel()
        sparse_model.fit_from_sparse_example_event_list(
            example_event_list=example_event_lists,
            known_dimensions=self.known_dimensions,
            n_epochs=200,
            lr=0.01,
            seed=123,
        )
        label_model = LabelModel(cardinality=self.known_dimensions.num_classes)
        label_model.fit(L, n_epochs=200, lr=0.01, seed=123)
        P_lm = label_model.get_conditional_probs()
        P_slm = sparse_model.get_conditional_probs()
        np.testing.assert_array_almost_equal(
            P_slm, P_lm,
        )
