"""
For compatibility with SuperGLUE tutorials, we we use these wrappers around
standard pytorch functional operators.
"""


import torch.nn.functional as F


def ce_loss(module_name, outputs, Y, active):
    # Subtract 1 from Ys to account for Snorkel reserving the label 0 for abstains
    # (but F.cross_entropy() expect 0-indexed labels)
    return F.cross_entropy(outputs[module_name][0][active], (Y.view(-1) - 1)[active])


def softmax(module_name, outputs):
    return F.softmax(outputs[module_name][0], dim=1)
