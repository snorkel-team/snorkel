import inspect
import pickle
from enum import Enum, auto
from typing import Any, Callable, List, Mapping, Optional

from snorkel.types import DataPoint, FieldMap


class MapperMode(Enum):
    """Enum defining mode for mapper depending on data point format."""

    NONE = auto()
    NAMESPACE = auto()
    PANDAS = auto()
    DASK = auto()
    SPARK = auto()


def get_parameters(
    f: Callable[..., Any], allow_args: bool = False, allow_kwargs: bool = False
) -> List[str]:
    """Get names of function parameters."""
    params = inspect.getfullargspec(f)
    if not allow_args and params[1] is not None:
        raise ValueError(f"Function {f.__name__} should not have *args")
    if not allow_kwargs and params[2] is not None:
        raise ValueError(f"Function {f.__name__} should not have **kwargs")
    return params[0]


class BaseMapper:
    """Base class for `Mapper` and `LambdaMapper`.

    Implements mode setting and deep copy functionality.

    Raises
    ------
    NotImplementedError
        Subclasses need to implement `_generate_mapped_data_point`
    """

    def _generate_mapped_data_point(self, x: DataPoint) -> DataPoint:
        raise NotImplementedError

    def set_mode(self, mode: MapperMode) -> None:
        """Change mapper mode, depending on data point format.

        Parameters
        ----------
        mode
            Mode to set mapper to
        """
        self.mode = mode

    def __call__(self, x: DataPoint) -> DataPoint:
        """Run mapping function on input data point.

        Deep copies the data point first so as not to make
        accidental in-place changes.

        Parameters
        ----------
        x
            Data point to run mapping function on

        Returns
        -------
        DataPoint
            Mapped data point of same format but possibly different fields
        """
        # NB: using pickle roundtrip as a more robust deepcopy
        # As an example, calling deepcopy on a pd.Series or SimpleNamespace
        # with a dictionary attribute won't create a copy of the dictionary
        x = pickle.loads(pickle.dumps(x))
        return self._generate_mapped_data_point(x)


class Mapper(BaseMapper):
    """Base class for any data point to data point mapping in the pipeline.

    Map data points to new data points by transforming, adding
    additional information, or decomposing into primitives. This module
    provides base classes for other operators like `TransformationFunction`
    and `Preprocessor`. We don't expect people to construct `Mapper`
    objects directly.

    A Mapper maps an data point to a new data point, possibly with
    a different schema. Subclasses of Mapper need to implement the
    `run` method, which takes fields of the data point as input
    and outputs new fields for the mapped data point as a dictionary.
    The `run` method should only be called internally by the `Mapper`
    object, not directly by a user.

    For an example of a Mapper, see
        `snorkel.labeling.preprocess.nlp.SpacyPreprocessor`

    Parameters
    ----------
    field_names
        A map from attribute names of the incoming data points
        to the input argument names of the `run` method. If None,
        the parameter names in the function signature are used.
    mapped_field_names
        A map from output keys of the `run` method to attribute
        names of the output data points. If None, the original
        output keys are used.

    Attributes
    ----------
    field_names
        See above
    mapped_field_names
        See above
    mode
        Mapper mode, corresponding to input data point format.
        See `MapperMode`.

    Raises
    ------
    NotImplementedError
        Subclasses must implement the `run` method
    ValueError
        Mapper mode must be set to a valid value
    """

    def __init__(
        self,
        field_names: Optional[Mapping[str, str]] = None,
        mapped_field_names: Optional[Mapping[str, str]] = None,
    ) -> None:
        if field_names is None:
            # Parse field names from `run(...)` if not provided
            field_names = {k: k for k in get_parameters(self.run)[1:]}
        self.field_names = field_names
        self.mapped_field_names = mapped_field_names
        self.mode = MapperMode.NONE

    def run(self, **kwargs: Any) -> Optional[FieldMap]:
        """Run the mapping operation using the input fields.

        The inputs to this function are fed by extracting the fields of
        the input data point using the keys of `field_names`. The output field
        names are converted using `mapped_field_names` and added to the
        data point.

        Returns
        -------
        Optional[FieldMap]
            A mapping from canonical output field names to their values.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method
        """
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
    """Define a mapper from a function.

    Convenience class for mappers that execute a simple
    function with no set up. The function should map from
    an input data point to a new data point directly, unlike
    `Mapper.run`. The original data point will not be updated,
    so in-place operations are safe.

    Parameters
    ----------
    f
        Function executing the mapping operation
    """

    def __init__(self, f: Callable[[DataPoint], Optional[DataPoint]]) -> None:
        self._f = f

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        return self._f(x)


def lambda_mapper(f: Callable[[DataPoint], Optional[DataPoint]]) -> LambdaMapper:
    """Decorate a function to define a LambdaMapper object.

    Parameters
    ----------
    f
        Function executing the mapping operation

    Example
    -------

    ```
    @lambda_mapper
    def concatenate_text(x: DataPoint) -> DataPoint:
        x.article = f"{title} {body}"
        return x

    isinstance(concatenate_text, LambdaMapper)  # true
    ```
    """
    return LambdaMapper(f=f)
