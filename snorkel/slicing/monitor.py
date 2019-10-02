import numpy as np
import pandas as pd

from snorkel.slicing import PandasSFApplier
from snorkel.slicing.sf import SlicingFunction


def slice_dataframe(
    df: pd.DataFrame, slicing_function: SlicingFunction
) -> pd.DataFrame:
    """Return a dataframe with examples corresponding to specified ``SlicingFunction``.

    Parameters
    ----------
    df
        A pandas DataFrame that will be sliced
    slicing_function
        SlicingFunction which will operate over df to return a subset of examples;
        function returns a subset of data for which ``slicing_function`` output is True

    Returns
    -------
    pd.DataFrame
        A DataFrame including only examples belonging to slice_name
    """

    S = PandasSFApplier([slicing_function]).apply(df)

    # Index into the SF labels by name
    df_idx = np.where(S[slicing_function.name])[0]  # type: ignore
    return df.iloc[df_idx]
