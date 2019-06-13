from enum import Enum, auto
from types import SimpleNamespace
from typing import Any, Mapping, NamedTuple, Union

from snorkel.types import Example, Field


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
        """Preprocess training examples by adding additional information
        or decomposing into primitives. A Preprocesser maps an example to
        a new example, possibly with a different schema.
        Subclasses of Preprocesser need to implement the `preprocess(...)`
        method, which takes fields of the training example as input
        and outputs new fields for the preprocessed training example.
        For an example of a preprocessor, see
            `snorkel.labeling.preprocess.nlp.SpacyPreprocessor`
        Args:
            * field_names: a map from attribute names of the incoming
                training examples to the input argument names of the
                `preprocess(...)` method
            * preprocessed_field_names: a map from output keys of the
                `preprocess(...)` method to attribute names of the
                output training examples
            * mode: a PreprocessorMode that specifies the type of the
                output training examples
        """
        self.field_names = field_names
        self.preprocessed_field_names = preprocessed_field_names
        self.mode = mode

    def set_mode(self, mode: PreprocessorMode) -> None:
        self.mode = mode

    def preprocess(self, **kwargs: Any) -> Mapping[str, Field]:
        raise NotImplementedError

    def __call__(self, example: Example) -> Example:
        field_map = {k: getattr(example, v) for k, v in self.field_names.items()}
        preprocessed_fields = self.preprocess(**field_map)
        preprocessed_fields = {
            v: preprocessed_fields[k] for k, v in self.preprocessed_field_names.items()
        }
        if self.mode == PreprocessorMode.NONE:
            raise ValueError(
                "No preprocessor mode set. Use `Preprocessor.set_mode(...)`."
            )
        if self.mode == PreprocessorMode.NAMESPACE:
            values = namespace_to_dict(example)
            values.update(preprocessed_fields)
            return SimpleNamespace(**values)
        if self.mode == PreprocessorMode.PANDAS:
            preprocessed_example = example.copy()
            for k, v in preprocessed_fields.items():
                preprocessed_example.loc[k] = v
            return preprocessed_example
        if self.mode == PreprocessorMode.DASK:
            raise NotImplementedError("Dask preprocessor mode not implemented")
        if self.mode == PreprocessorMode.SPARK:
            raise NotImplementedError("Spark preprocessor mode not implemented")
        else:
            raise ValueError(
                f"Preprocessor mode {self.mode} not recognized. Options: {PreprocessorMode}."
            )
