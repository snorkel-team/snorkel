from typing import Iterable, NamedTuple, Tuple

import numpy as np

CliqueSet = Iterable[
    int,
]
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


class UnnormalizedObjective(Exception):
    """Raised when an Objective matrix has values outside of [0,1]

    """

    pass


class KnownDimensions(NamedTuple):
    r"""The known dimensions for the problem we are solving.

    Parameters
    ----------
    num_functions
        The number of labeling functions (corresponds to LabelModel.m)
    num_classes
        The number of classes in the problem, excluding "Abstain" (corresponds to LabelModel.cardinality)
    num_examples
        The number of examples that the data was created with. (Corresponds to model.n). This is needed to
        normalize the Objective matrix
    """
    num_functions: int
    num_classes: int
    num_examples: int

    @property
    def num_events(self) -> int:
        """How many indicator random variables do we have (1 per event)
        """
        return self.num_functions * self.num_classes


class ExampleEventListOccurence(NamedTuple):
    event_ids: Iterable[int]


class EventCooccurence(NamedTuple):
    event_a: int
    event_b: int
    frequency: int
