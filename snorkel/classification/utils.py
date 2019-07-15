import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

TensorCollection = Union[torch.Tensor, dict, list, tuple]


def list_to_tensor(item_list: List[torch.Tensor]) -> torch.Tensor:
    """Convert a list of torch.Tensor into a single torch.Tensor."""

    # Convert single value tensor
    if all(item_list[i].dim() == 0 for i in range(len(item_list))):
        item_tensor = torch.stack(item_list, dim=0)
    # Convert 2 or more-D tensor with the same shape
    elif all(
        (item_list[i].size() == item_list[0].size()) and (len(item_list[i].size()) != 1)
        for i in range(len(item_list))
    ):
        item_tensor = torch.stack(item_list, dim=0)
    # Convert reshape to 1-D tensor and then convert
    else:
        item_tensor, _ = pad_batch([item.view(-1) for item in item_list])

    return item_tensor


def pad_batch(
    batch: List[torch.Tensor],
    max_len: int = 0,
    pad_value: int = 0,
    left_padded: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert the batch into a padded tensor and mask tensor.

    Parameters
    ----------
    batch
        The data for padding
    max_len
        Max length of sequence of padding
    pad_value
        The value to use for padding
    left_padded
        If True, pad on the left, otherwise on the right

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The padded matrix and correspoing mask matrix.
    """

    batch_size = len(batch)
    max_seq_len = int(np.max([len(item) for item in batch]))  # type: ignore

    if max_len > 0 and max_len < max_seq_len:
        max_seq_len = max_len

    padded_batch = batch[0].new_full((batch_size, max_seq_len), pad_value)

    for i, item in enumerate(batch):
        length = min(len(item), max_seq_len)  # type: ignore
        if left_padded:
            padded_batch[i, -length:] = item[-length:]
        else:
            padded_batch[i, :length] = item[:length]

    mask_batch = torch.eq(padded_batch.clone().detach(), pad_value).type_as(
        padded_batch
    )

    return padded_batch, mask_batch


def recursive_merge_dicts(
    x: dict, y: dict, misses: str = "report", verbose: Optional[int] = None
) -> dict:
    """Merge dictionary y into a copy of x."""

    def recurse(x: dict, y: dict, misses: str = "report", verbose: int = 1) -> bool:
        found = True
        for k, v in y.items():
            found = False
            if k in x:
                found = True
                if isinstance(x[k], dict):
                    if not isinstance(v, dict):
                        msg = f"Attempted to overwrite dict {k} with " f"non-dict: {v}"
                        raise ValueError(msg)
                    # If v is {}, set x[k] = {} instead of recursing on empty dict
                    # Otherwise, recurse on the items in v
                    if v:
                        recurse(x[k], v, misses, verbose)
                    else:
                        x[k] = v
                else:
                    if x[k] == v:
                        msg = f"Reaffirming {k}={x[k]}"
                    else:
                        msg = f"Overwriting {k}={x[k]} to {k}={v}"
                        x[k] = v
                    if verbose > 1 and k != "verbose":
                        print(msg)
            else:
                for kx, vx in x.items():
                    if isinstance(vx, dict):
                        found = recurse(vx, {k: v}, misses="ignore", verbose=verbose)
                    if found:
                        break
            if not found:
                msg = f'Could not find kwarg "{k}" in destination dict.'
                if misses == "insert":
                    x[k] = v
                    if verbose > 1:
                        print(f"Added {k}={v} from second dict to first")
                elif misses == "exception":
                    raise ValueError(msg)
                elif misses == "report":
                    print(msg)
                else:
                    pass
        return found

    # If verbose is not provided, look for an value in y first, then x
    # (Do this because 'verbose' kwarg is often inside one or both of x and y)
    if verbose is None:
        verbose = y.get("verbose", x.get("verbose", 1))
        assert isinstance(verbose, int)

    z = copy.deepcopy(x)
    recurse(z, y, misses, verbose)
    return z


def move_to_device(
    obj: TensorCollection, device: int = -1
) -> TensorCollection:  # pragma: no cover
    """Recursively move torch.Tensors to a given CUDA device.

    Given a structure (possibly) containing Tensors on the CPU, move all the Tensors
    to the specified GPU (or do nothing, if they should beon the CPU).

    Originally from:
    https://github.com/HazyResearch/metal/blob/mmtl_clean/metal/utils.py

    Paramters
    ---------
    obj
        Tensor or collection of Tensors to move
    device
        Device to move Tensors to
        device = -1 -> "cpu"
        device =  0 -> "cuda:0"
    """

    if device < 0 or not torch.cuda.is_available():
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(device)  # type: ignore
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, device) for item in obj])
    else:
        return obj


def collect_flow_outputs_by_suffix(
    flow_dict: Dict[str, torch.Tensor], suffix: str
) -> List[torch.Tensor]:
    """Return flow_dict outputs specified by suffix, ordered by sorted flow_name."""
    return [
        flow_dict[flow_name][0]
        for flow_name in sorted(flow_dict.keys())
        if flow_name.endswith(suffix)
    ]
