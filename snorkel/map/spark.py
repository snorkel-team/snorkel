from pyspark.sql import Row

from snorkel.types import FieldMap

from .core import Mapper


def _update_fields(x: Row, mapped_fields: FieldMap) -> Row:
    # ``pyspark.sql.Row`` objects are not mutable, so need to
    # reconstruct
    all_fields = x.asDict()
    all_fields.update(mapped_fields)
    return Row(**all_fields)


def make_spark_mapper(mapper: Mapper) -> Mapper:
    """Convert ``Mapper`` to be compatible with PySpark.

    Parameters
    ----------
    mapper
        Mapper to make compatible with PySpark
    """
    mapper._update_fields = _update_fields  # type: ignore
    return mapper
