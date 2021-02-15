from typing import List, Optional

import numpy as np
from scipy.sparse import csr_matrix

from snorkel.labeling.model.sparse_label_model.base_sparse_label_model import BaseSparseLabelModel
from snorkel.labeling.model.sparse_label_model.sparse_label_model_helpers import  KnownDimensions, \
    ExampleEventListOccurence


class SparseExampleEventListLabelModel(BaseSparseLabelModel):
    def fit_from_sparse_example_event_list(
        self,
        example_event_list: List[ExampleEventListOccurence],
        known_dimensions: KnownDimensions,
        Y_dev: Optional[np.ndarray] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs
    ):
        objective = self._prepare_objective_from_sparse_example_eventlist(
            example_events_list=example_event_list, known_dimensions=known_dimensions
        )
        return self.fit_from_objective(
            objective=objective,
            known_dimensions=known_dimensions,
            Y_dev=Y_dev,
            class_balance=class_balance,
            **kwargs,
        )
    @staticmethod
    def _prepare_objective_from_sparse_example_eventlist(known_dimensions :KnownDimensions, example_events_list : List[ExampleEventListOccurence]):

        L_index = SparseExampleEventListLabelModel.get_l_ind(example_events_list, known_dimensions)
        objective = (L_index.T @L_index) /known_dimensions.num_examples
        return objective.todense()

    @staticmethod
    def get_l_ind(example_events_list, known_dimensions,return_array=False):
        L_index = csr_matrix((known_dimensions.num_examples, known_dimensions.num_events))
        for num, example in enumerate(example_events_list):
            for event_id in example.event_ids:
                L_index[num, event_id] = 1
        if return_array:
            return L_index.toarray()
        return L_index
