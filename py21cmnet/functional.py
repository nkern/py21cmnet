"""
functional module
"""

import numpy as np
import torch
from torch import nn


class ModifiedTanh(nn.Module):
    r"""Modified Tanh - i.e. a sharper sigmoid

    :math:`\text{MTanh}(x) = \text{Tanh}(x) / 2 + 0.5`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    def __init__(self):
        super(ModifiedTanh, self).__init__()

    def forward(self, input):
        return torch.tanh(input) / 2 + 0.5


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
