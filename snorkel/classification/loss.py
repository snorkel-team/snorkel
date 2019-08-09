from typing import List, Mapping, Optional

import torch
import torch.nn.functional as F

Outputs = Mapping[str, List[torch.Tensor]]


def cross_entropy_with_probs(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints

    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.

    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.

    Parameters
    ----------
    input
        A [num_points, num_classes] tensor of logits
    target
        A [num_points, num_classes] tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses

    Returns
    -------
    torch.Tensor
        The calculated loss

    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted
    """
    num_points, num_classes = input.shape
    # Note that t.new_zeros, t.new_full put tensor on same device as t
    cum_losses = input.new_zeros(num_points)
    for y in range(num_classes):
        target_temp = input.new_full((num_points,), y, dtype=torch.long)
        y_loss = F.cross_entropy(input, target_temp, reduction="none")
        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


def cross_entropy_with_probs_from_outputs(
    op_name: str, outputs: Outputs, Y: torch.Tensor, active: torch.IntTensor
) -> torch.Tensor:
    """Calculate mean cross-entropy loss for active outputs using probabilistic labels

    Parameters
    ----------
    op_name
        The name of the operation whose output should be used for calculating loss
    outputs
        The dictionary of operation outputs
    Y
        The gold labels (integers) to calculate loss from
    active
        A mask of which data points to consider when calculating loss

    Returns
    -------
    torch.Tensor
        The calculated loss
    """
    return cross_entropy_with_probs(outputs[op_name][0][active], Y[active])


def cross_entropy_from_outputs(
    op_name: str, outputs: Outputs, Y: torch.Tensor, active: torch.IntTensor
) -> torch.Tensor:
    """Calculate mean cross-entropy loss for the active outputs of the specified op.

    Note: TBD
    ...gold labels must be integers. To use "soft" probabilistic labels, use cross_entropy_with_probs() instead.

    Parameters
    ----------
    op_name
        The name of the operation whose output should be used for calculating loss
    outputs
        The dictionary of operation outputs
    Y
        The gold labels (integers) to calculate loss from
    active
        A mask of which data points to consider when calculating loss

    Returns
    -------
    torch.Tensor
        The calculated loss
    """
    return F.cross_entropy(outputs[op_name][0][active], (Y.view(-1))[active])
