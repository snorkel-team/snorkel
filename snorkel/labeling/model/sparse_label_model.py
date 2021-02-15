# -*- coding: utf-8 -*-
"""Sparse Data Helpers

Indexing throughout this module is 0 based, with the assumption that "abstains" are ommited. 

When working with larger datasets, it can be convenient to load the data in sparse format. This module
provides utilities to do so. We provide functions for a number of cases. 

The user has the AugmentedMatrix (L_ind) in tuple form. AugmentedMatrix is of shape (num_examples,numfuncs*num_classes) 
and the user has a list of tuples (i,j) that indicate that event j occoured for example i. 

The user has a list of 3-tuples(i,j,k) such that for document i, labeling function j predicted class k.

The user has a list of 3-tuples (i,j,c) where i and j range over [0,num_funcs*num_classes] such that 
the events  i and j were observed to have co-occur c times. 

The user has a list of 3-tuples (i,j,f) where i and j range over [0,num_funcs*num_classes] such that 
the events  i and j co-occur with frequency f where f is in (0,1]

"""
from snorkel.labeling.model.label_model import LabelModel, TrainConfig
from typing import List, Tuple, Iterable, Dict, Optional, Union, Any, NamedTuple
from scipy.sparse import csr_matrix
import numpy as np
import torch
from snorkel.utils import probs_to_preds

CliqueSet = Iterable[int,]
"""
    A cliqueset is a variable length tuple. An instance of a Cliqueset represents the event ids that co-occured (e.g. a
    clique). Where an event_id is defined as fund_id*num_labels+label_id
"""
CliqueSetList = Iterable[CliqueSet]

CliqueSetProbs = Tuple[CliqueSetList, np.ndarray]
"""
     CliqueSetProbs is a tuple whose first element is a list of CliqueSets and second element is an array of 
     probabiltiies  returned from the label model such that the probabilities at index i of the array correspond
     to the clique at index i of the CliqueSetList. 
     This could be a dict, but we want to preserve the whole array to leverage the pre-existing code for calculating
     a predicted class
"""
CliqueSetProbsAndPreds = Tuple[CliqueSetList, np.ndarray, np.ndarray]
"""
    CliqueSetProbsAndPreds extends CliqueSetProbs with a third tuple element, an array of predicted class_ids 
"""


class SparseLabelModel(LabelModel):
    def __init__(self, known_dimensions: Optional[KnownDimensions] = None):
        super().__init__()
        if known_dimensions:
            self._set_constants(known_dimensions=known_dimensions)
    def get_weights(self) -> np.ndarray:
        raise NotImplementedError("SparseLabelModel doesn't support get_weights")

    def predict(
        self,
        cliquesets: CliqueSetList,
        return_probs: Optional[bool] = False,
        tie_break_policy: str = "abstain",
    ) -> Union[CliqueSetProbs, CliqueSetProbsAndPreds]:
        #the users cliqueset might be an unordered iterable (set) so we take the ordered list
        cliqsets_list,Y_probs = self.predict_proba_from_cliqueset(cliquesets)
        if return_probs:
            Y_p = probs_to_preds(Y_probs, tie_break_policy)
            result : CliqueSetProbsAndPreds = (cliqsets_list,Y_probs,Y_p)
        else:
            result : CliqueSetProbs = (cliqsets_list,Y_probs)
        return result


    def predict_proba_from_cliqueset(
        self, cliquesets: CliqueSetList
    ) -> CliqueSetProbs:
        """
            This function can make inference many orders of magnitude faster for larger datasets.

            In the data representation of L_ind where each row is a document and each column corresponds to an event "
            function x predicted class y", the 1s on L_ind essentially define a fully connected graph, or cliqueset.
            while their are num_classes^num_functions possible cliquesets, in practice we'll see a very small subset of
            those.
            In our exerpiments, where num_functions=40 and num_classes=3 we observed 600 cliquesets whereas 3^40 were possible.

            This function receives a trained model, and a list of cliquesets (indexed by event_id "func_id*num_labels+label_id")
            loads those in a sparse format and returns to predictions keyed by cliqueset



        """
        rows = []
        cols = []
        data = []
        cliquesets_list =[] #We rehold the cliquesets in a list, because the input might be an unorderable set
        for num, cs in enumerate(cliquesets):
            cliquesets_list.append(cs)
            for event_id in cs:
                rows.append(num)
                cols.append(event_id)
                data.append(1)
        sparse_input_l_ind = csr_matrix((data, (rows, cols)), shape=(len(cliquesets), self.d))
        predicted_probs = self.predict_proba(
            sparse_input_l_ind.todense(), is_augmented=True
        )
        result: CliqueSetProbs = (cliquesets_list, predicted_probs)

        return result

    def fit(
        self,
        L_train: np.ndarray,
        Y_dev: Optional[np.ndarray] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            #Using the function __name__ in an f-string makes refactoring/renaming easier
            f"SparseLabelModel doesn't support calls to fit. Please use {self.fit_from_objective.__name__} or {self.fit_from_sparse_indicators.__name__}"
        )

    def fit_from_objective(
        self, objective: np.ndarray, known_dimensions: KnownDimensions,         Y_dev: Optional[np.ndarray] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs
    ):
        self._set_constants(known_dimensions=known_dimensions)
        self.O = torch.from_numpy(objective)
        self._common_training_preamble(Y_dev=Y_dev,class_balance=class_balance,**kwargs)
        self._common_training_loop()

    def fit_from_sparse_indicators(
        self,
        sparse_event_cooccurence: List[Tuple[int, int, int]],
        known_dimensions: KnownDimensions,
            Y_dev: Optional[np.ndarray] = None,
            class_balance: Optional[List[float]] = None,
            **kwargs
    ):
        objective = self._prepare_objective_from_sparse_event_cooccurence(
            sparse_event_cooccurence, known_dimensions
        )
        return self.fit_from_objective(
            objective=objective, known_dimensions=known_dimensions,
            Y_dev=Y_dev, class_balance=class_balance, **kwargs
        )
    @staticmethod
    def _prepare_objective_from_sparse_event_cooccurence(

        sparse_event_cooccurence: List[Tuple[int, int, int]],
        known_dimensions: KnownDimensions,
    ):
        sparse_L_ind = SparseLabelModel._prepare_sparse_L_ind(
            known_dimensions, sparse_event_cooccurence
        )
        objective = (sparse_L_ind.T @ sparse_L_ind) / known_dimensions.num_examples
        return objective.todense()

    @staticmethod
    def _prepare_sparse_L_ind(known_dimensions, sparse_event_cooccurence):
        rows = []
        cols = []
        data = []
        for (row, col, count) in sparse_event_cooccurence:
            rows.append(row)
            cols.append(col)
            data.append(count)
        rows = np.array(rows)
        cols = np.array(cols)
        sparse_L_ind = csr_matrix(
            (data, (rows, cols),),  # Notice that this is a tuple with a tuple
            shape=(known_dimensions.num_examples, known_dimensions.num_events),
        )
        return sparse_L_ind


class KnownDimensions(NamedTuple):
    num_functions: int
    num_classes: int
    num_examples: Optional[int]

    @property
    def num_events(self):
        """
            How many indicator random variables do we have (1 per event)
        """
        return self.num_functions * self.num_classes