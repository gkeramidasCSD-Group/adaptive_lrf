"""Small building blocks shared by more than one architecture family (see
arch/<network>.py for everything else - each network's baseline and
adaptive classes live together in their own file)."""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth: during training, zeroes a residual branch for a
    random subset of the batch (with probability `drop_prob`) and rescales
    the rest. Identity in eval mode.

    Input: (batch, ...) tensor. Output: same shape."""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], 1, 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        return x / keep_prob * random_tensor
