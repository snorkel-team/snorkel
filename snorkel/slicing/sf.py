"""Define SF appliers as simple alias of LF counterparts for now"""
from typing import Callable

from snorkel.labeling.lf import LabelingFunction, labeling_function


class SlicingFunction(LabelingFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, label_space=(0, 1))


class slicing_function(labeling_function):
    def __call__(self, f: Callable[..., int]) -> SlicingFunction:
        name = self.name or f.__name__
        return SlicingFunction(
            name=name,
            f=f,
            schema=self.schema,
            resources=self.resources,
            preprocessors=self.preprocessors,
            fault_tolerant=self.fault_tolerant,
        )
