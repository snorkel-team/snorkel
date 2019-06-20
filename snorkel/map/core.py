import inspect
from enum import Enum, auto
from types import SimpleNamespace
from typing import Any, Callable, List, Mapping, NamedTuple, Optional, Union

from snorkel.types import DataPoint, FieldMap


class MapperMode(Enum):
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


def get_parameters(
    f: Callable[..., Any], allow_args: bool = False, allow_kwargs: bool = False
) -> List[str]:
    params = inspect.getfullargspec(f)
    if not allow_args and params[1] is not None:
        raise ValueError(f"Function {f.__name__} should not have *args")
    if not allow_kwargs and params[2] is not None:
        raise ValueError(f"Function {f.__name__} should not have **kwargs")
    return params[0]


class Mapper:
    def __init__(
        self,
        field_names: Optional[Mapping[str, str]] = None,
        mapped_field_names: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Map data points to new data points by transforming, adding
        additional information, or decomposing into primitives. This module
        provides base classes for other operators like `TransformationFunction`
        and `Preprocessor`. We don't expect people to construct `Mapper`
        objects directly.

        A Mapper maps an data point to a new data point, possibly with
        a different schema. Subclasses of Mapper need to implement the
        `run(...)` method, which takes fields of the data point as input
        and outputs new fields for the mapped data point as a dictionary.
        For an example of a Mapper, see
            `snorkel.labeling.preprocess.nlp.SpacyPreprocessor`
        Args:
            * field_names: a map from attribute names of the incoming
                data points to the input argument names of the
                `run(...)` method. If None, the parameter names in the
                function signature are used.
            * mapped_field_names: a map from output keys of the
                `run(...)` method to attribute names of the
                output data points. If None, the original output
                keys are used.
        """
        if field_names is None:
            # Parse field names from `run(...)` if not provided
            field_names = {k: k for k in get_parameters(self.run)[1:]}
        self.field_names = field_names
        self.mapped_field_names = mapped_field_names
        self.mode = MapperMode.NONE

    def set_mode(self, mode: MapperMode) -> None:
        self.mode = mode

    def run(self, **kwargs: Any) -> FieldMap:
        raise NotImplementedError

    def __call__(self, x: DataPoint) -> DataPoint:
        field_map = {k: getattr(x, v) for k, v in self.field_names.items()}
        mapped_fields = self.run(**field_map)
        assert isinstance(mapped_fields, dict)
        if self.mapped_field_names is not None:
            mapped_fields = {
                v: mapped_fields[k] for k, v in self.mapped_field_names.items()
            }
        if self.mode == MapperMode.NONE:
            raise ValueError("No Mapper mode set. Use `Mapper.set_mode(...)`.")
        if self.mode == MapperMode.NAMESPACE:
            values = namespace_to_dict(x)
            values.update(mapped_fields)
            return SimpleNamespace(**values)
        if self.mode == MapperMode.PANDAS:
            x_mapped = x.copy()
            for k, v in mapped_fields.items():
                x_mapped.loc[k] = v
            return x_mapped
        if self.mode == MapperMode.DASK:
            raise NotImplementedError("Dask Mapper mode not implemented")
        if self.mode == MapperMode.SPARK:
            raise NotImplementedError("Spark Mapper mode not implemented")
        else:
            raise ValueError(
                f"Mapper mode {self.mode} not recognized. Options: {MapperMode}."
            )


class LambdaMapper(Mapper):
    def __init__(self, f: Callable[..., FieldMap]) -> None:
        """Convenience class for Mappers that execute a simple
        function with no set up. The function arguments are parsed
        to determine the input field names of the data points.

        Args:
            * f: the function executing the mapping operation
        """
        self._f = f
        field_names = {k: k for k in get_parameters(f)}
        super().__init__(field_names=field_names, mapped_field_names=None)

    def run(self, **kwargs: Any) -> FieldMap:
        return self._f(**kwargs)


def lambda_mapper(f: Callable[..., FieldMap]) -> LambdaMapper:
    """Decorator to define a LambdaMapper object from a function

        Example usage:

        ```
        @lambda_mapper()
        def concatenate_text(title: str, body: str) -> FieldMap:
            return dict(article=f"{title} {body}")

        isinstance(concatenate_text, Mapper)  # true
        ```
        """
    return LambdaMapper(f=f)
