from pyspark.sql import Row

from snorkel.types import FieldMap

from .core import Mapper


class SparkMapper(Mapper):
    """Base class for any `Mapper` that runs on PySpark `Row` objects.

    See `snorkel.map.core.Mapper`.

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
    memoize
        Memoize mapper outputs?

    Raises
    ------
    NotImplementedError
        Subclasses must implement the `run` method
    """

    def _update_fields(self, x: Row, mapped_fields: FieldMap) -> Row:
        # `pyspark.sql.Row` objects are not mutable, so need to
        # reconstruct
        all_fields = x.asDict()
        all_fields.update(mapped_fields)
        return Row(**all_fields)
