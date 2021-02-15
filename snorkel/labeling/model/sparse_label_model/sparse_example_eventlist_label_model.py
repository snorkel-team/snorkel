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
    def fit_from_sparse_example_event_list(
        self,
        example_event_list: List[ExampleEventListOccurence],
        known_dimensions: KnownDimensions,
        Y_dev: Optional[np.ndarray] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs: Any
    ) -> None:
        '''
        """Train label model on a list of ExampleEventOccourrences

        """Train label model.

        Train label model to estimate mu, the parameters used to combine LFs.

        Parameters
        ----------
        example_event_list
            A list of n examples as NamedTuples, with each named tuple containing a list of the event ids that
            occoured for that example
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
        This is a useful class when you can get the data in a structure that maps docs to event_ids. A pseudo-sql example

            select doc_id,array_agg(function_id*num_labels+class_id) as event_ids
            from prediction
            group by doc_id

        Examples
        --------
        """
            The following example demonstrates converting a standard LabelModel format into the data format this
            class expects, and then training it.

        """
        L # Is our label matrix, what you'd typically pass to label_model.fit(L)

        example_event_lists: List[ExampleEventListOccurence] = []

        for example_num,example in enumerate(L):
                event_list =[]
                for func_id,cls_id in enumerate(example):
                    if(cls_id)>-1:
                        event_id = func_id*self.known_dimensions.num_classes+cls_id
                        event_list.append(event_id)
                example_event_lists.append((ExampleEventListOccurence(event_list)))

        sparse_model = SparseExampleEventListLabelModel()
        sparse_model.fit_from_sparse_example_event_list(example_event_list=example_event_lists,
                                                 known_dimensions=self.known_dimensions,
                                                 n_epochs=200, lr=0.01, seed=123
                                                 )


        '''
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
    ) -> None:

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
