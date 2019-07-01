import pandas as pd
from tqdm import tqdm

from snorkel.augmentation.tf import TransformationFunctionMode

from .tf_applier import BaseTFApplier


class PandasTFApplier(BaseTFApplier):
    """TF applier for a Pandas DataFrame.

    Data points are stored as Series in a DataFrame. The TFs
    run on data points obtained via a `pandas.DataFrame.iterrows`
    call, which is single-process and can be slow for large DataFrames.
    For large datasets, consider `DaskTFApplier` or `SparkTFApplier`.
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        """Augment a Pandas DataFrame of data points using TFs and policy.

        Parameters
        ----------
        df
            Pandas DataFrame containing data points to be transformed

        Returns
        -------
        pd.DataFrame
            Augmented DataFrame of data points
        """
        self._set_tf_mode(TransformationFunctionMode.PANDAS)
        x_transformed = []
        for _, x in tqdm(df.iterrows(), total=len(df)):
            x_transformed.extend(self._apply_policy_to_data_point(x))
        return pd.concat(x_transformed, axis=1).T
