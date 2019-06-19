"""
For compatibility with SuperGLUE tutorials, we we use these wrappers around
standard pytorch functional operators.
TODO: Redesign MTL model construction so that regular F.cross_entropy and
F.softmax can be used directly.
"""


import torch.nn.functional as F


def ce_loss(module_name, outputs, Y, active):
    return F.cross_entropy(outputs[module_name][0][active], (Y.view(-1) - 1)[active])


def softmax(module_name, outputs):
    return F.softmax(outputs[module_name][0], dim=1)
