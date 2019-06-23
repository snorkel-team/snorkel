import pandas as pd
from tqdm import tqdm

from snorkel.augmentation.tf import TransformationFunctionMode

from .tf_applier import BaseTFApplier


class PandasTFApplier(BaseTFApplier):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        self._set_tf_mode(TransformationFunctionMode.PANDAS)
        x_transformed = []
        for _, x in tqdm(df.iterrows(), total=len(df)):
            x_transformed.extend(self._apply_policy_to_data_point(x))
        return pd.concat(x_transformed, axis=1).T
