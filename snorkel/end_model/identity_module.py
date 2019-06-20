import torch.nn as nn


class IdentityModule(nn.Module):
    """A default identity input module that simply passes the input through."""

    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, x):
        return x
