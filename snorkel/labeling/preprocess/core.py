from enum import Enum, auto
from types import SimpleNamespace
from typing import Any, Mapping, NamedTuple, Union

from snorkel.types import DataPoint, FieldMap


class PreprocessorMode(Enum):
    NONE = auto()
    NAMESPACE = auto()
    PANDAS = auto()
    DASK = auto()
    SPARK = auto()


def namespace_to_dict(x: Union[SimpleNamespace, NamedTuple]) -> dict:
    """Convert a SimpleNamespace or NamedTuple to a dict"""
    if isinstance(x, SimpleNamespace):
        return vars(x)
    return x._asdict()


class Preprocessor:
    def __init__(
        self,
        field_names: Mapping[str, str],
        preprocessed_field_names: Mapping[str, str],
        mode: PreprocessorMode = PreprocessorMode.NONE,
    ) -> None:
        """Preprocess data points by adding additional information
        or decomposing into primitives. A Preprocesser maps an data point to
        a new data point, possibly with a different schema.
        Subclasses of Preprocesser need to implement the `preprocess(...)`
        method, which takes fields of the data point as input
        and outputs new fields for the preprocessed data point.
        For an data point of a preprocessor, see
            `snorkel.labeling.preprocess.nlp.SpacyPreprocessor`
        Args:
            * field_names: a map from attribute names of the incoming
                data points to the input argument names of the
                `preprocess(...)` method
            * preprocessed_field_names: a map from output keys of the
                `preprocess(...)` method to attribute names of the
                output data points
            * mode: a PreprocessorMode that specifies the type of the
                output data point
        """
        self.field_names = field_names
        self.preprocessed_field_names = preprocessed_field_names
        self.mode = mode

    def set_mode(self, mode: PreprocessorMode) -> None:
        self.mode = mode

    def preprocess(self, **kwargs: Any) -> FieldMap:
        raise NotImplementedError

    def __call__(self, x: DataPoint) -> DataPoint:
        field_map = {k: getattr(x, v) for k, v in self.field_names.items()}
        preprocessed_fields = self.preprocess(**field_map)
        preprocessed_fields = {
            v: preprocessed_fields[k] for k, v in self.preprocessed_field_names.items()
        }
        if self.mode == PreprocessorMode.NONE:
            raise ValueError(
                "No preprocessor mode set. Use `Preprocessor.set_mode(...)`."
            )
        if self.mode == PreprocessorMode.NAMESPACE:
            values = namespace_to_dict(x)
            values.update(preprocessed_fields)
            return SimpleNamespace(**values)
        if self.mode == PreprocessorMode.PANDAS:
            x_preprocessed = x.copy()
            for k, v in preprocessed_fields.items():
                x_preprocessed.loc[k] = v
            return x_preprocessed
        if self.mode == PreprocessorMode.DASK:
            raise NotImplementedError("Dask preprocessor mode not implemented")
        if self.mode == PreprocessorMode.SPARK:
            raise NotImplementedError("Spark preprocessor mode not implemented")
        else:
            raise ValueError(
                f"Preprocessor mode {self.mode} not recognized. Options: {PreprocessorMode}."
            )
