# -*- coding: utf-8 -*-
"""Sparse Data Helpers.

Indexing throughout this module is 0 based, with the assumption that "abstains" are ommited.

When working with larger datasets, it can be convenient to load the data in sparse format. This module
provides utilities to do so. We provide functions for a number of cases.

Case 1:
    The user has the AugmentedMatrix (L_ind) in tuple form. AugmentedMatrix is of shape
    (num_examples,numfuncs*num_classes)
    and the user has a list of tuples (i,j) that indicate that event j occoured for example i.

Case 2:
    The user has a list of 3-tuples(i,j,k) such that for document i, labeling function j predicted class k.

The Case 3:
    user has a list of 3-tuples (i,j,c) where i and j range over [0,num_funcs*num_classes] such that
    the events  i and j were observed to have co-occur c times.

Case 5:
    The user has a list of 3-tuples (i,j,f) where i and j range over [0,num_funcs*num_classes] such that
    the events  i and j co-occur with frequency f where f is in (0,1]

"""
from typing import Any, List, Optional, Union

import numpy as np
import torch
from scipy.sparse import csr_matrix

from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling.model.sparse_label_model.sparse_label_model_helpers import (
    CliqueSetList,
    CliqueSetProbs,
    CliqueSetProbsAndPreds,
    KnownDimensions,
    UnnormalizedObjectiveException,
)
from snorkel.utils import probs_to_preds


class BaseSparseLabelModel(LabelModel):
    """A ```LabelModel``` that accepts sparse formatted inputs for training and prediction."""

    def __init__(self, known_dimensions: Optional[KnownDimensions] = None):
        super().__init__()
        if known_dimensions:
            self._set_constants(known_dimensions=known_dimensions)

    def get_weights(self) -> np.ndarray:
        r"""Not implemented. A ```SparseLabelModel``` doesn't support this method.

        We need to calculate coverage from a sparse format which is not implemented.

        Raises
        ------
        NotImplementedError
            This method is not implemented

        """
        raise NotImplementedError("SparseLabelModel doesn't support get_weights")

    def predict(
        self,
        cliquesets: CliqueSetList,
        return_probs: Optional[bool] = False,
        tie_break_policy: str = "abstain",
    ) -> Union[CliqueSetProbs, CliqueSetProbsAndPreds]:
        r"""Run prediction on a ```CliqueSetList```.

        A ```LabelModel's``` output is determined by the "Events" that cooccured, which we call a CliqueSet.
        This accepts an iterable of CliqueSets and runs prediction on each. In practice, the number of unique CliqueSets
        present in the data set is order of magnitude smaller than the total number of CliqueSets as well the number of
        distinct examples. Hence, this method runs inference once per inputed CliqueSet and returns the CliqueSets as well
        as the predections, allowing the user to join back on the original data at reduced computational cost.

        Parameters
        ----------
        cliquesets
            An iterable of CliqueSets
        return_probs
            Whether to return probs along with preds
        tie_break_policy
            Policy to break ties when converting probabilistic labels to predictions

        Returns
        -------
        CliqueSetProbs
            A 2-tuple whose first element is a list of CliqueSets and whose second element is an [len(cliquesets),k] array
            such that ar[i] are the probabilities for cliqueset i

        CliqueSetProbsAndPreds
            A 3-tuple whose first element is a list of CliqueSets and whose second element is an [len(cliquesets),k] array,
            and third element as [len(cliquestes),1] array such that ar_1[i] are the probabilities for cliqueset i
            and ar_2[i] is the predicted class for that cliqueset.

        """
        # The users cliqueset might be an unordered iterable (set) so we take the ordered list
        cliqsets_list, Y_probs = self.predict_proba_from_cliqueset(cliquesets)
        if return_probs:
            Y_p = probs_to_preds(Y_probs, tie_break_policy)
            return (cliqsets_list, Y_probs, Y_p)
        else:
            return (cliqsets_list, Y_probs)

    def predict_proba_from_cliqueset(self, cliquesets: CliqueSetList) -> CliqueSetProbs:
        r"""Return label probabilities P(Y | \lambda).

            This function can make inference many orders of magnitude faster for larger datasets.

            In the data representation of L_ind where each row is a document and each column corresponds to an event "
            function x predicted class y", the 1s on L_ind essentially define a fully connected graph, or cliqueset.
            while their are num_classes^num_functions possible cliquesets, in practice we'll see a very small subset of
            those.
            In our exerpiments, where num_functions=40 and num_classes=3 we observed 600 cliquesets whereas 3^40 were possible.

            This function receives a trained model, and a list of cliquesets (indexed by event_id "func_id*num_labels+label_id")
            loads those in a sparse format and returns to predictions keyed by cliqueset

        Parameters
        ----------
        cliquesets
            An iterable of CliqueSets

        Returns
        -------
        CliqueSetProbs
            A 2-tuple whose first element is a list of CliqueSets and whose second element is an [len(cliquesets),k] array
            such that ar[i] are the probabilities for cliqueset i
        """
        rows = []
        cols = []
        data = []
        cliquesets_list = (
            []
        )  # We rehold the cliquesets in a list, because the input might be an unorderable set
        count = 0
        for num, cs in enumerate(cliquesets):
            count += 1
            cliquesets_list.append(cs)
            for event_id in cs:
                rows.append(num)
                cols.append(event_id)
                data.append(1)
        sparse_input_l_ind = csr_matrix((data, (rows, cols)), shape=(count, self.d))
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
        """Not implemented.

        Raises
        ------
        NotImplementedError
            This method is not implemented

        """
        raise NotImplementedError(
            "BaseSparseLabelModel does not support training. Use one of the derived classes and their explicit fit "
            "functions"
        )

    def fit_from_objective(
        self,
        objective: np.ndarray,
        known_dimensions: KnownDimensions,
        Y_dev: Optional[np.ndarray] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs: Any
    ) -> None:
        """Fits the LabelModel to a given Objective matrix.

        Parameters
        ----------
        objective
            a [num_events,num_events] matrix with values in [0,1]
        known_dimensions
            The known dimensions of the problem as a ```KnownDimensions``` named tuple.

        Example
        -------
        known_dimensions = KnownDimensions(
            num_classes=3, num_examples=5, num_functions=7
        )

        possible_function_values = np.eye(known_dimensions.num_classes)
        choice_set = np.random.choice(
            known_dimensions.num_classes,
            size=[known_dimensions.num_functions, known_dimensions.num_examples],
        )
        L_ind = np.hstack(possible_function_values[choice_set])
        objective = (L_ind.T @ cls.L_ind) / known_dimensions.num_examples

        sparse_model = SparseLabelModel()
        sparse_model.fit_from_objective(objective)

        Notes
        -----
            When calling this function directly, don't forget to normalize your objective by num_examples.
            If you see very large losses and lack of convergence, you may have forgotten to do so.

        Raises
        ------
        UnnormalizedObjective
            Raises an exception if the objective has values outside [0,1]

        """
        if not np.all((objective <= 1) & (objective >= 0)):
            raise UnnormalizedObjectiveException(
                "The objective function you passed in has values outside [0,1]. Did you forget to normalize by num_examples ? "
            )
        self._set_config_and_seed(**kwargs)
        self._set_constants(known_dimensions=known_dimensions)
        self.O = torch.from_numpy(objective)
        self._training_preamble(Y_dev=Y_dev, class_balance=class_balance, **kwargs)
        self._training_loop()
