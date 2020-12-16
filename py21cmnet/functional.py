"""
functional module
"""

import numpy as np
import torch
from torch import nn


class ModifiedSigmoid(nn.Module):
    r"""Modified Sigmoid - i.e. a sharper sigmoid

    :math:`\text{MSigmoid}(x) = 1 / (1 + \exp(-x * c))`

    Args:
        c : float, coefficient of input in exponential

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    def __init__(self, c=1):
        super(ModifiedSigmoid, self).__init__()
        self.c = c

    def forward(self, input):
        return 1 / (1 + torch.exp(-input * self.c))

    def extra_repr(self):
        return 'c={}'.format(self.c)


class ModifiedHardtanh(nn.Module):
    r"""Modified Hardtanh - i.e. a sharper Hard sigmoid

    :math:`\text{MHardtanh}(x) = \text{Hardtanh}(x) / 2 + 0.5`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    def __init__(self):
        super(ModifiedHardtanh, self).__init__()

    def forward(self, input):
        return nn.functional.hardtanh(input) / 2 + 0.5

