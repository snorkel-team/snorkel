import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceCombinerModule(nn.Module):
    """A module for combining the weighted representations learned by slices"""

    def __init__(
        self,
        slice_ind_key="_ind",
        slice_pred_key="_pred",
        slice_pred_feat_key="_pred_transform",
    ):
        super().__init__()

        self.slice_ind_key = slice_ind_key
        self.slice_pred_key = slice_pred_key
        self.slice_pred_feat_key = slice_pred_feat_key

    def forward(self, outputs):
        # Gather names of slice heads (both indicator and predictor heads)
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

        slice_feat_names = sorted(
            [
                flow_name
                for flow_name in outputs.keys()
                if self.slice_pred_feat_key in flow_name
            ]
        )

        slice_representations = torch.cat(
            [
                outputs[slice_feat_name][0].unsqueeze(1)
                for slice_feat_name in slice_feat_names
            ],
            dim=1,
        )

        A = (
            F.softmax(indicator_preds * predictor_preds, dim=1)
            .unsqueeze(-1)
            .expand([-1, -1, slice_representations.size(-1)])
        )

        reweighted_rep = torch.sum(A * slice_representations, 1)

        return reweighted_rep
