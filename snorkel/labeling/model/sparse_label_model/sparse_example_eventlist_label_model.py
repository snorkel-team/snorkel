from typing import Any, List, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix

from snorkel.labeling.model.sparse_label_model.base_sparse_label_model import (
    BaseSparseLabelModel,
)
from snorkel.labeling.model.sparse_label_model.sparse_label_model_helpers import (
    ExampleEventListOccurence,
    KnownDimensions,
)


class SparseExampleEventListLabelModel(BaseSparseLabelModel):
    """A  subclass```LabelModel``` that trains on a list of Event Coocurrences per example"""
    def fit_from_sparse_example_event_list(
        self,
        example_event_list: List[ExampleEventListOccurence],
        known_dimensions: KnownDimensions,
        Y_dev: Optional[np.ndarray] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs: Any
    ) -> None:
        """Train label model from a list of Event Coocurrences per example.
        Train label model to estimate mu, the parameters used to combine LFs.

        Parameters
        ----------
        example_event_list
            A list of ```ExampleEventListOccurence```
        known_dimensions
            The known dimensions of the problem
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
        
        Examples
        --------
            known_dimensions = KnownDimensions(
                num_classes=7, num_examples=1000, num_functions=10
            )
            np.random.seed(123)
            P, Y, L = generate_simple_label_matrix(
                known_dimensions.num_examples,
                known_dimensions.num_functions,
                known_dimensions.num_classes,
            )
            sparse_event_occurence: List[EventCooccurence] = []
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
                known_dimensions=known_dimensions,
                n_epochs=200,
                lr=0.01,
                seed=123,
            )


        """
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
    def _prepare_objective_from_sparse_example_eventlist(
        known_dimensions: KnownDimensions,
        example_events_list: List[ExampleEventListOccurence],
    ) -> np.ndarray:

        L_index = SparseExampleEventListLabelModel.get_l_ind(
            example_events_list, known_dimensions
        )
        objective = (L_index.T @ L_index) / known_dimensions.num_examples
        return objective.todense()

    @staticmethod
    def get_l_ind(
        example_events_list: List[ExampleEventListOccurence],
        known_dimensions: KnownDimensions,
        return_array: bool = False,
    ) -> Union[csr_matrix, np.ndarray]:
        """
        Calculates the L_ind matrix, in a sparse format by default.
        We separate this out for easier testing.

        set return_array to true to get a numpy array isntead of a csr_matrix
        """
        L_index = csr_matrix(
            (known_dimensions.num_examples, known_dimensions.num_events)
        )
        for num, example in enumerate(example_events_list):
            for event_id in example.event_ids:
                L_index[num, event_id] = 1
        if return_array:
            return L_index.toarray()
        return L_index
