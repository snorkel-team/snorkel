from typing import Dict, List

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

    def forward(  # type:ignore
        self, flow_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Reweights and combines predictor representations given output dict.

        Parameters
        ----------
        flow_dict
            A dict of data fields from slicing task flow containing specific keys
            from indicator ops, pred ops, and pred transform ops (slice_ind_key,
            slice_pred_key, slice_pred_feat_key) for each slice.

            NOTE: The flow_dict outputs for the ind/pred heads must be raw logits.

        Returns
        -------
        torch.Tensor
            The reweighted predictor representation
        """

        # Concatenate indicator head predictions into tensor [batch_size x num_slices]
        indicator_outputs = self._collect_flow_outputs_by_key(
            flow_dict, self.slice_ind_key
        )
        indicator_preds = torch.cat(
            [
                F.softmax(output, dim=1)[:, 0].unsqueeze(1)
                for output in indicator_outputs
            ],
            dim=-1,
        )

        # Concatenate predictor head confidences into tensor [batch_size x num_slices]
        predictor_outputs = self._collect_flow_outputs_by_key(
            flow_dict, self.slice_pred_key
        )

        predictor_confidences = torch.cat(
            [
                # Compute the "confidence" using the max score across classes
                torch.max(F.softmax(output, dim=1), dim=1)[0].unsqueeze(1)
                for output in predictor_outputs
            ],
            dim=-1,
        )

        # Concatenate each predictor feature (to be combined into reweighted
        # representation) into [batch_size x 1 x feat_dim] tensor
        predictor_feat_outputs = self._collect_flow_outputs_by_key(
            flow_dict, self.slice_pred_feat_key
        )
        slice_representations = torch.cat(
            [output.unsqueeze(1) for output in predictor_feat_outputs], dim=1
        )

        # Attention weights used to combine each of the slice_representations
        # incorporates the indicator (whether we are in the slice or not) and
        # predictor (confidence of a learned slice head)
        A = F.softmax(indicator_preds * predictor_confidences, dim=1)

        # Match dims [bach_size x num_slices x feat_dim] of slice_representations
        A = A.unsqueeze(-1).expand([-1, -1, slice_representations.size(-1)])

        # Reweight representations with weighted sum across slices
        reweighted_rep = torch.sum(A * slice_representations, dim=1)
        return reweighted_rep

    def _collect_flow_outputs_by_key(
        self, flow_dict: Dict[str, torch.Tensor], key: str
    ) -> List[torch.Tensor]:
        """Return flow_dict outputs specified by key, ordered by sorted flow_name."""
        return [
            flow_dict[flow_name][0]
            for flow_name in sorted(flow_dict.keys())
            if key in flow_name
        ]
