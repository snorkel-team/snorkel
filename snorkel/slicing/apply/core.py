from snorkel.labeling import LFApplier, PandasLFApplier


class SFApplier(LFApplier):
    """SF applier for a list of data points.

    See ``snorkel.labeling.core.LFApplier`` for details.
    """

    _use_recarray = True


class PandasSFApplier(PandasLFApplier):
    """SF applier for a Pandas DataFrame.

    See ``snorkel.labeling.core.PandasLFApplier`` for details.
    """

    _use_recarray = True
