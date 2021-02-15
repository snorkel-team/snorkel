from typing import List, Optional

import numpy as np

from snorkel.labeling.model.sparse_label_model.base_sparse_label_model import BaseSparseLabelModel
from snorkel.labeling.model.sparse_label_model.sparse_label_model_helpers import EventCooccurence, KnownDimensions


class SparseEventPairLabelModel(BaseSparseLabelModel):
    def fit_from_sparse_event_cooccurrence(
        self,
        sparse_event_occurence: List[EventCooccurence],
        known_dimensions: KnownDimensions,
        Y_dev: Optional[np.ndarray] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs
    ):
        objective = self._prepare_objective_from_sparse_event_cooccurence(known_dimensions, sparse_event_occurence)
        return self.fit_from_objective(
            objective=objective,
            known_dimensions=known_dimensions,
            Y_dev=Y_dev,
            class_balance=class_balance,
            **kwargs,
        )
    @staticmethod
    def _prepare_objective_from_sparse_event_cooccurence(known_dimensions, sparse_event_occurence : List[EventCooccurence]):
        objective = np.zeros(shape=[known_dimensions.num_events, known_dimensions.num_events])
        for co_oc in sparse_event_occurence:
            objective[co_oc.event_a, co_oc.event_b] = co_oc.frequency / known_dimensions.num_examples
            objective[co_oc.event_b, co_oc.event_a] = co_oc.frequency / known_dimensions.num_examples
        return objective

