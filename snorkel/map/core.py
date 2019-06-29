import inspect
import pickle
from collections import Hashable
from enum import Enum, auto
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

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


def is_hashable(obj: Any) -> bool:
    """Test if object is hashable via duck typing.

    NB: not using `collections.Hashable` as some objects
    (e.g. pandas.Series) have a `__hash__` method to throw
    a more specific exception.
    """
    try:
        hash(obj)
        return True
    except Exception:
        return False


def get_hashable(obj: Any) -> Hashable:
    """Get a hashable version of a potentially unhashable object.

    This helper is used for caching mapper outputs of data points.
    For common data point formats (e.g. SimpleNamespace, pandas.Series),
    produces hashable representations of the values using a `frozenset`.
    For objects like `pandas.Series`, the name/index indentifier is dropped.

    Parameters
    ----------
    obj
        Object to get hashable version of

    Returns
    -------
    Hashable
        Hashable representation of object values

    Raises
    ------
    ValueError
        No hashable proxy for object
    """
    # If hashable already, just return
    if is_hashable(obj):
        return obj
    # Get dictionary from SimpleNamespace
    if isinstance(obj, SimpleNamespace):
        obj = vars(obj)
    # For dictionaries or pd.Series, construct a frozenset from items
    # Also recurse on values in case they aren't hashable
    if isinstance(obj, (dict, pd.Series)):
        return frozenset((k, get_hashable(v)) for k, v in obj.items())
    # For lists, recurse on values
    if isinstance(obj, (list, tuple)):
        return tuple(get_hashable(v) for v in obj)
    # For NumPy arrays, hash the byte representation of the data array
    if isinstance(obj, np.ndarray):
        return obj.data.tobytes()
    raise ValueError(f"Object {obj} has no hashing proxy.")


class BaseMapper:
    """Base class for `Mapper` and `LambdaMapper`.

    Implements mode setting, memoization, and deep copy functionality.

    Raises
    ------
    NotImplementedError
        Subclasses need to implement `_generate_mapped_data_point`
    """

    def __init__(self, memoize: bool) -> None:
        self.memoize_outputs(memoize)
        self.reset_cache()

    def memoize_outputs(self, mode: bool) -> None:
        """Set memoization mode.

        Parameters
        ----------
        mode
            Memoize mapper outputs?
        """
        self._memoize_outputs = mode

    def reset_cache(self) -> None:
        """Reset the memoization cache."""
        self._cache: Dict[DataPoint, DataPoint] = {}

    def set_mode(self, mode: MapperMode) -> None:
        """Change mapper mode, depending on data point format.

        Parameters
        ----------
        mode
            Mode to set mapper to
        """
        self.mode = mode

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        raise NotImplementedError

    def __call__(self, x: DataPoint) -> Optional[DataPoint]:
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
        if self._memoize_outputs:
            # NB: don't do `self._cache.get(...)` first in case cached value is `None`
            x_hashable = get_hashable(x)
            if x_hashable in self._cache:
                return self._cache[x_hashable]
        # NB: using pickle roundtrip as a more robust deepcopy
        # As an example, calling deepcopy on a pd.Series or SimpleNamespace
        # with a dictionary attribute won't create a copy of the dictionary
        x = pickle.loads(pickle.dumps(x))
        x_mapped = self._generate_mapped_data_point(x)
        if self._memoize_outputs:
            self._cache[x_hashable] = x_mapped
        return x_mapped


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
    memoize
        Memoize mapper outputs?

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
        memoize: bool = False,
    ) -> None:
        if field_names is None:
            # Parse field names from `run(...)` if not provided
            field_names = {k: k for k in get_parameters(self.run)[1:]}
        self.field_names = field_names
        self.mapped_field_names = mapped_field_names
        self.mode = MapperMode.NONE
        super().__init__(memoize)

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
        if self.mode in (MapperMode.NAMESPACE, MapperMode.PANDAS, MapperMode.DASK):
            for k, v in mapped_fields.items():
                setattr(x, k, v)
            return x
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
    memoize
        Memoize mapper outputs?
    """

    def __init__(
        self, f: Callable[[DataPoint], Optional[DataPoint]], memoize: bool = False
    ) -> None:
        self._f = f
        super().__init__(memoize)

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        return self._f(x)


class lambda_mapper:
    """Decorate a function to define a LambdaMapper object.

    Parameters
    ----------
    memoize
        Memoize mapper outputs?

    Example
    -------

    ```
    @lambda_mapper()
    def concatenate_text(x: DataPoint) -> DataPoint:
        x.article = f"{title} {body}"
        return x

    isinstance(concatenate_text, LambdaMapper)  # true
    ```
    """

    def __init__(self, memoize: bool = False) -> None:
        self.memoize = memoize

    def __call__(self, f: Callable[[DataPoint], Optional[DataPoint]]) -> LambdaMapper:
        """Wrap a function to create a `LambdaMapper`.

        Parameters
        ----------
        f
            Function executing the mapping operation

        Returns
        -------
        LambdaMapper
            New `LambdaMapper` executing operation in wrapped function
        """
        return LambdaMapper(f=f, memoize=self.memoize)
