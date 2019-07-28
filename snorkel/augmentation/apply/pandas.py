from typing import List

import pandas as pd
from tqdm import tqdm

from .core import BaseTFApplier


class PandasTFApplier(BaseTFApplier):
    """TF applier for a Pandas DataFrame.

    Data points are stored as Series in a DataFrame. The TFs
    run on data points obtained via a ``pandas.DataFrame.iterrows``
    call, which is single-process and can be slow for large DataFrames.
    For large datasets, consider ``DaskTFApplier`` or ``SparkTFApplier``.
    """

    def apply_generator(self, df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
        """Augment a Pandas DataFrame of data points using TFs and policy in batches.

        This method acts as a generator, yielding augmented data points for
        a given input batch of data points. This can be useful in a training
        loop when it is too memory-intensive to pregenerate all transformed
        examples.

        Parameters
        ----------
        df
            Pandas DataFrame containing data points to be transformed
        batch_size
            Batch size for generator. Yields augmented data points
            for the next ``batch_size`` input data points.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame of data points in augmented data set
        """
        batch_transformed: List[pd.Series] = []
        for i, (_, x) in enumerate(df.iterrows()):
            batch_transformed.extend(self._apply_policy_to_data_point(x))
            if (i + 1) % batch_size == 0:
                yield pd.concat(batch_transformed, axis=1).T.infer_objects()
                batch_transformed = []
        yield pd.concat(batch_transformed, axis=1).T.infer_objects()

    def apply(self, df: pd.DataFrame, progress_bar: bool = True) -> pd.DataFrame:
        """Augment a Pandas DataFrame of data points using TFs and policy.

        Parameters
        ----------
        df
            Pandas DataFrame containing data points to be transformed
        progress_bar
            Display a progress bar?

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame of data points in augmented data set
        """
        x_transformed: List[pd.Series] = []
        for _, x in tqdm(df.iterrows(), total=len(df), disable=(not progress_bar)):
            x_transformed.extend(self._apply_policy_to_data_point(x))
        return pd.concat(x_transformed, axis=1).T.infer_objects()
