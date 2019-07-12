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
    slice_ind_key
        Suffix of operation corresponding to the slice indicator heads
    slice_pred_key
        Suffix of operation corresponding to the slice predictor heads
    slice_pred_feat_key
        Suffix of operation corresponding to the slice predictor features heads

    Attributes
    ----------
    slice_ind_key
        See above
    slice_pred_key
        See above
    slice_pred_feat_key
        See above
    """

    def __init__(
        self,
        slice_ind_key: str = "_ind_head",
        slice_pred_key: str = "_pred_head",
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
            A dict of data fields from slicing task flow containing specific keys
            from indicator ops, pred ops, and pred transform ops (slice_ind_key,
            slice_pred_key, slice_pred_feat_key) for each slice

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
        predictor_confidences = torch.cat(
            [
                F.softmax(
                    # Compute the "confidence" using the max score across classes
                    torch.max(outputs[slice_pred_name][0], dim=1).values
                ).unsqueeze(1)
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
        # incorporates the indicator (whether we are in the slice or not) and
        # predictor (confidence of a learned slice head)
        A = torch.softmax(indicator_preds * predictor_confidences, dim=1)

        # Expand weights and match dims [bach_size x num_slices x feat_dim] of slice_representations
        A = A.unsqueeze(-1).expand([-1, -1, slice_representations.size(-1)])

        # Reweight representations with weighted sum across slices
        reweighted_rep = torch.sum(A * slice_representations, dim=1)
        return reweighted_rep
