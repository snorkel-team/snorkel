import inspect
import pickle
from enum import Enum, auto
from typing import Any, Callable, List, Mapping, Optional

from snorkel.types import DataPoint, FieldMap


class MapperMode(Enum):
    NONE = auto()
    NAMESPACE = auto()
    PANDAS = auto()
    DASK = auto()
    SPARK = auto()


def get_parameters(
    f: Callable[..., Any], allow_args: bool = False, allow_kwargs: bool = False
) -> List[str]:
    params = inspect.getfullargspec(f)
    if not allow_args and params[1] is not None:
        raise ValueError(f"Function {f.__name__} should not have *args")
    if not allow_kwargs and params[2] is not None:
        raise ValueError(f"Function {f.__name__} should not have **kwargs")
    return params[0]


class BaseMapper:
    def _generate_mapped_data_point(self, x: DataPoint) -> DataPoint:
        raise NotImplementedError

    def set_mode(self, mode: MapperMode) -> None:
        self.mode = mode

    def __call__(self, x: DataPoint) -> DataPoint:
        # NB: using pickle roundtrip as a more robust deepcopy
        # As an example, calling deepcopy on a pd.Series or SimpleNamespace
        # with a dictionary attribute won't create a copy of the dictionary
        x = pickle.loads(pickle.dumps(x))
        return self._generate_mapped_data_point(x)


class Mapper(BaseMapper):
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

    def run(self, **kwargs: Any) -> Optional[FieldMap]:
        raise NotImplementedError

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        field_map = {k: getattr(x, v) for k, v in self.field_names.items()}
        mapped_fields = self.run(**field_map)
        if mapped_fields is None:
            return None
        assert isinstance(mapped_fields, dict)
        if self.mapped_field_names is not None:
            mapped_fields = {
                v: mapped_fields[k] for k, v in self.mapped_field_names.items()
            }
        if self.mode == MapperMode.NONE:
            raise ValueError("No Mapper mode set. Use `Mapper.set_mode(...)`.")
        if self.mode in (MapperMode.NAMESPACE, MapperMode.PANDAS):
            for k, v in mapped_fields.items():
                setattr(x, k, v)
            return x
        if self.mode == MapperMode.DASK:
            raise NotImplementedError("Dask Mapper mode not implemented")
        if self.mode == MapperMode.SPARK:
            raise NotImplementedError("Spark Mapper mode not implemented")
        else:
            raise ValueError(
                f"Mapper mode {self.mode} not recognized. Options: {MapperMode}."
            )


class LambdaMapper(BaseMapper):
    def __init__(self, f: Callable[[DataPoint], Optional[DataPoint]]) -> None:
        """Convenience class for Mappers that execute a simple
        function with no set up. The function should map from
        an input DataPoint to a new DataPoint. The original DataPoint
        will not be updated, so in-place operations are safe.

        Args:
            * f: the function executing the mapping operation
        """
        self._f = f

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        return self._f(x)


def lambda_mapper(f: Callable[[DataPoint], Optional[DataPoint]]) -> LambdaMapper:
    """Decorator to define a LambdaMapper object from a function

        Example usage:

        ```
        @lambda_mapper()
        def concatenate_text(x: DataPoint) -> DataPoint:
            x.article = f"{title} {body}"
            return x

        isinstance(concatenate_text, LambdaMapper)  # true
        ```
        """
    return LambdaMapper(f=f)
