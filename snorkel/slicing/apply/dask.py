from snorkel.labeling.apply.dask import (  # pragma: no cover
    DaskLFApplier,
    PandasParallelLFApplier,
)


class DaskSFApplier(DaskLFApplier):  # pragma: no cover
    """SF applier for a Dask DataFrame.

    See ``snorkel.labeling.apply.dask.DaskLFApplier`` for details.
    """

    _use_recarray = True


class PandasParallelSFApplier(PandasParallelLFApplier):  # pragma: no cover
    """Parallel SF applier for a Pandas DataFrame.

    See ``snorkel.labeling.apply.dask.PandasParallelLFApplier`` for details.
    """

    _use_recarray = True
