from snorkel.labeling.apply.dask import DaskLFApplier, PandasParallelLFApplier


class DaskSFApplier(DaskLFApplier):  # pragma: no cover
    """SF applier for a Dask DataFrame.

    See `snorkel.labeling.apply.dask.DaskLFApplier` for details.
    """
    pass


class ParallelPandasSFApplier(PandasParallelLFApplier):  # pragma: no cover
    """Parallel SF applier for a Pandas DataFrame.

    See `snorkel.labeling.apply.dask.PandasParallelLFApplier` for details.
    """
    pass
