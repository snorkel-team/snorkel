"""
Commonly used loss and output functions for Task objects
# TODO: move filtering of actives and pulling out modules before these functions
"""


import torch.nn.functional as F


def ce_loss(module_name, outputs, Y, active):
    # Subtract 1 from hard labels in Y to account for Snorkel reserving the label 0 for
    # abstains while F.cross_entropy() expects 0-indexed labels
    return F.cross_entropy(outputs[module_name][0][active], (Y.view(-1) - 1)[active])


def softmax(module_name, outputs):
    return F.softmax(outputs[module_name][0], dim=1)
