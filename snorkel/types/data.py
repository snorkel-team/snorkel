from typing import Any, Mapping, Sequence, NamedTuple, Optional

DataPoint = Any
DataPoints = Sequence[DataPoint]

Field = Any
FieldMap = Mapping[str, Field]
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
