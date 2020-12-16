"""
functional testing suite
"""
import numpy as np
import torch
import os
import yaml

from py21cmnet import functional


def test_activations():
    x = np.linspace(-10, 10, 100)
    ms = functional.ModifiedSigmoid(c=2)(torch.as_tensor(x)).detach().numpy()
    mht = functional.ModifiedHardtanh()(torch.as_tensor(x)).detach().numpy()
    assert np.isclose(ms.min(), 0)
    assert np.isclose(ms.max(), 1)
    assert np.isclose(mht.min(), 0)
    assert np.isclose(mht.max(), 1)
