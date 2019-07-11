from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceCombinerModule(nn.Module):
    """A module for combining the weighted representations learned by slices.

    Intended for use with task flow including:
        * Indicator operations
        * Prediction operations
        * Prediction transform features

    Parameters
    ----------
    slice_ind_key : optional
        Suffix of operation corresponding to the slice indicator heads
    slice_pred_key : optional
        Suffix of operation corresponding to the slice predictor heads
    slice_pred_feat_key : optional
        Suffix of operation corresponding to the slice predictor features heads
    """

    def __init__(
        self,
        slice_ind_key: str = "_ind",
        slice_pred_key: str = "_pred",
        slice_pred_feat_key: str = "_pred_transform",
    ) -> None:
        super().__init__()

        self.slice_ind_key = slice_ind_key
        self.slice_pred_key = slice_pred_key
        self.slice_pred_feat_key = slice_pred_feat_key

    def forward(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        """Reweights and combines predictor representations given output dict.

        Parameters
        ----------
        outputs
            A dict of data fields containing indicator, predictor, and predictor
            transform ops

        Returns
        -------
        torch.Tensor
            The reweighted predictor representation
        """

        # Gather names of slice heads (both indicator and predictor heads)
        # This provides a static ordering by which to index into the 'outputs' dict
        slice_ind_op_names = sorted(
            [
                flow_name
                for flow_name in outputs.keys()
                if self.slice_ind_key in flow_name
            ]
        )
        slice_pred_op_names = sorted(
            [
                flow_name
                for flow_name in outputs.keys()
                if self.slice_pred_key in flow_name
            ]
        )

        # Concatenate the predictions from the predictor head/indicator head
        # into a [batch_size x num_slices] tensor
        indicator_preds = torch.cat(
            [
                F.softmax(outputs[slice_ind_name][0])[:, 0].unsqueeze(1)
                for slice_ind_name in slice_ind_op_names
            ],
            dim=-1,
        )
        predictor_preds = torch.cat(
            [
                F.softmax(outputs[slice_pred_name][0])[:, 0].unsqueeze(1)
                for slice_pred_name in slice_pred_op_names
            ],
            dim=-1,
        )

        # Collect names of predictor "features" that will be combined into the final
        # reweighted representation
        slice_feat_names = sorted(
            [
                flow_name
                for flow_name in outputs.keys()
                if self.slice_pred_feat_key in flow_name
            ]
        )

        # Concatenate each predictor feature into [batch_size x 1 x feat_dim] tensor
        slice_representations = torch.cat(
            [
                outputs[slice_feat_name][0].unsqueeze(1)
                for slice_feat_name in slice_feat_names
            ],
            dim=1,
        )

        # Attention weights used to combine each of the slice_representations
        A = (
            # Combine the indicator (whether we are in the slice or not) and
            # predictor (confidence of a learned slice head) as attention weights
            F.softmax(indicator_preds * predictor_preds, dim=1)
            # Match the dimensions of the slice_representations
            .unsqueeze(-1).expand([-1, -1, slice_representations.size(-1)])
        )

        # Reweight representations by class Sum across all classes
        reweighted_rep = torch.sum(A * slice_representations, dim=1)
        return reweighted_rep
