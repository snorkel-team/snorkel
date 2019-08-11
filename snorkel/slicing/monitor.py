from typing import Dict, List, Union

import numpy as np
import pandas as pd

from snorkel.analysis.scorer import Scorer
from snorkel.labeling.lf import LabelingFunction
from snorkel.slicing import PandasSFApplier

SliceMetricsDict = Dict[str, Dict[str, float]]


class SliceScorer:
    """Scorer that returns metrics on overall performance, as well as slices.

    Parameters
    ----------
    scorer
        A pre-defined ``Scorer`` used for evaluation
    slice_names
        A list of slice names corresponding to columns of ``S_matrix`` (accepted by score method)

    Attributes
    ----------
    scorer
        See above
    slice_names
        See above.
    """

    def __init__(self, scorer: Scorer, slice_names: List[str]):
        self.scorer = scorer
        self.slice_names = slice_names

    def score(
        self,
        S: np.ndarray,
        golds: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray,
        as_dataframe: bool = False,
    ) -> Union[SliceMetricsDict, pd.DataFrame]:
        """Calculate user-specified and/or user-defined metrics overall + slices.

        Parameters
        ----------
        S
            A [num_examples x num_slices] matrix of slicing function outputs
        golds
            Gold (aka ground truth) labels (integers)
        preds
            Predictions (integers)
        probs:
            Probabilities (floats)
        as_dataframe
            A boolean indicating whether to return results as pandas DataFrame (True)
            or dict (False)

        Returns
        -------
        Union[SliceMetricsDict, pd.DataFrame]
            A dictionary mapping slice_name to metric names to metric scores
            or aforementioned dictionary formatted as pandas DataFrame
        """
        assert S.shape[1] == len(self.slice_names)
        assert S.shape[0] == len(golds) == len(preds) == len(probs)

        # Include overall metrics
        metrics_dict = dict()
        metrics_dict.update({"overall": self.scorer.score(golds, preds, probs)})

        # Include slice metrics
        for idx, slice_name in enumerate(self.slice_names):
            mask = S[:, idx].astype(bool)
            metrics_dict.update(
                {slice_name: self.scorer.score(golds[mask], preds[mask], probs[mask])}
            )

        if as_dataframe:
            return pd.DataFrame.from_dict(metrics_dict).transpose()
        else:
            return metrics_dict


def slice_dataframe(
    df: pd.DataFrame, slicing_function: LabelingFunction
) -> pd.DataFrame:
    """Return a dataframe with examples corresponding to specified SlicingFunction.

    Parameters
    ----------
    df
        A pandas DataFrame that will be sliced
    slicing_function
        SlicingFunction which will operate over df to return a subset of examples;
        function returns a subset of data for which slicing_function output is True

    Returns
    -------
    pd.DataFrame
        A DataFrame including only examples belonging to slice_name
    """

    S = PandasSFApplier([slicing_function]).apply(df)

    # Index into the SF labels for the first (and only) column
    df_idx = np.where(S[:, 0])[0]
    return df.iloc[df_idx]
