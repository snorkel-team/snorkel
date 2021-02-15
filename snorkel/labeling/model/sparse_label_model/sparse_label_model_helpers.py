from typing import Iterable, Tuple, NamedTuple, Optional

import numpy as np

CliqueSet = Iterable[
    int,
]
CliqueSetList = Iterable[CliqueSet]
CliqueSetProbs = Tuple[CliqueSetList, np.ndarray]
CliqueSetProbsAndPreds = Tuple[CliqueSetList, np.ndarray, np.ndarray]


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
    num_examples: Optional[int]

    @property
    def num_events(self):
        """How many indicator random variables do we have (1 per event)
        """
        return self.num_functions * self.num_classes


class ExampleEventListOccurence(NamedTuple):
    event_ids :Iterable[int]


class EventCooccurence(NamedTuple):
    event_a :int
    event_b :int
    frequency : int
