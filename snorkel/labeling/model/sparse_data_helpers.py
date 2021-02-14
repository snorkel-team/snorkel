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
from snorkel.labeling.model.label_model import LabelModel
from typing import List,  Tuple, Iterable, Dict
from scipy.sparse import csr_matrix
import numpy as np
import torch
from snorkel.types.data import KnownDimensions


def predict_probs_from_cliqueset(
    trained_model: LabelModel, cliqueset_indice_list: Iterable[Iterable[int]]
):
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
    for num, cs in enumerate(cliqueset_indice_list):
        for event_id in cs:
            rows.append(num)
            cols.append(event_id)
            data.append(1)
    sparse_input_l_ind = csr_matrix(
        (data, (rows, cols)), shape=(len(rows), trained_model.d)
    )
    predicted_probs = trained_model.predict_proba(sparse_input_l_ind.todense(),is_augmented=True)
    result_dict: Dict[tuple, np.array] ={}
    for cs, probs in zip(cliqueset_indice_list, predicted_probs):
        result_dict[tuple(cs)] = probs.tolist()[0]
    return result_dict


def train_model_from_known_objective(
    objective: np.array, known_dimensions: KnownDimensions, **kwargs
):
    model = LabelModel(cardinality=known_dimensions.num_classes, **kwargs)
    model._set_constants(known_dimensions=known_dimensions)
    model.O = torch.from_numpy(objective)
    model._common_training_preamble()
    model._common_training_loop()
    return model


def train_model_from_sparse_event_cooccurence(
    sparse_event_cooccurence: List[Tuple[int, int, int]],
    known_dimensions: KnownDimensions,
):
    objective = _prepare_objective_from_sparse_event_cooccurence(
        sparse_event_cooccurence, known_dimensions
    )
    return train_model_from_known_objective(
        objective=objective, known_dimensions=known_dimensions
    )


def _prepare_objective_from_sparse_event_cooccurence(
    sparse_event_cooccurence: List[Tuple[int, int, int]],
    known_dimensions: KnownDimensions,
):
    sparse_L_ind = _prepare_sparse_L_ind(known_dimensions, sparse_event_cooccurence)
    objective = (sparse_L_ind.T @ sparse_L_ind) / known_dimensions.num_examples
    return objective.todense()


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
