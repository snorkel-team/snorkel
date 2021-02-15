from typing import Any, List, Optional

import numpy as np

from snorkel.labeling.model.sparse_label_model.base_sparse_label_model import (
    BaseSparseLabelModel,
)
from snorkel.labeling.model.sparse_label_model.sparse_label_model_helpers import (
    EventCooccurence,
    KnownDimensions,
)


class SparseEventPairLabelModel(BaseSparseLabelModel):
    def fit_from_sparse_event_cooccurrence(
        self,
        sparse_event_occurence: List[EventCooccurence],
        known_dimensions: KnownDimensions,
        Y_dev: Optional[np.ndarray] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs: Any
    ) -> None:
        '''
        """Train label model on a list of EventCooccurence.

        If you have data in the form event_a,event_b,frequency then use this class (on large datasets). It skips
        calculating L_ind and the serialization cost of moving a lot of data.



        Parameters
        ----------
        sparse_event_occurence
            A list of 3-tuples such that t[0] = event_a t[1]=event_b and t[2] is how many times they co-occured.
        Y_dev
            Gold labels for dev set for estimating class_balance, by default None
        class_balance
            Each class's percentage of the population, by default None
        **kwargs
            Arguments for changing train config defaults.

            n_epochs
                The number of epochs to train (where each epoch is a single
                optimization step), default is 100
            lr
                Base learning rate (will also be affected by lr_scheduler choice
                and settings), default is 0.01
            l2
                Centered L2 regularization strength, default is 0.0
            optimizer
                Which optimizer to use (one of ["sgd", "adam", "adamax"]),
                default is "sgd"
            optimizer_config
                Settings for the optimizer
            lr_scheduler
                Which lr_scheduler to use (one of ["constant", "linear",
                "exponential", "step"]), default is "constant"
            lr_scheduler_config
                Settings for the LRScheduler
            prec_init
                LF precision initializations / priors, default is 0.7
            seed
                A random seed to initialize the random number generator with
            log_freq
                Report loss every this many epochs (steps), default is 10
            mu_eps
                Restrict the learned conditional probabilities to
                [mu_eps, 1-mu_eps], default is None

        Raises
        ------
        Exception
            If loss in NaN

        Notes
        -----
        This is a useful class when you can get the data in a structure that that g. A pseudo-sql example

        with doc_events as (
            select doc_id, function_id * num_labels + class_id as event_id
            from prediction
        )
        select a.event_id,b.event_id,count(*) freq from doc_events a inner join doc_events b
        on a.doc_id = b.doc_id and a.doc_id >=b.doc_id
        group by a.doc_id,b.doc_id
        '''

    @staticmethod
    def _prepare_objective_from_sparse_event_cooccurence(
        known_dimensions: KnownDimensions,
        sparse_event_occurence: List[EventCooccurence],
    ) -> np.ndarray:
        objective = np.zeros(
            shape=[known_dimensions.num_events, known_dimensions.num_events]
        )
        for co_oc in sparse_event_occurence:
            objective[co_oc.event_a, co_oc.event_b] = (
                co_oc.frequency / known_dimensions.num_examples
            )
            objective[co_oc.event_b, co_oc.event_a] = (
                co_oc.frequency / known_dimensions.num_examples
            )
        return objective
