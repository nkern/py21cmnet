"""
utils module
"""
import numpy as np


class Roll:
    """roll a box along last ndim axes"""
    def __init__(self, shift=None, ndim=3):
        """roll a periodic box by "shift" pixels
        Args:
            shift : int or tuple
                If roll a box by shift pixels along
                the last ndim axes. Default is random.
            ndim : int
                Dimensionality of the box
        """
        self.shift = shift
        self.ndim = ndim

    def __call__(self, box):
        shift = self.shift
        if shift is None:
            shift = np.random.randint(0, box.shape[-1], self.ndim)
        if self.ndim == 2;
            return np.roll(box, shift, axis=(-1, -2))
        elif self.ndim == 3:
            return np.roll(box, shift, axis=(-1, -2))


class DownSample:
    """down sample a box along last ndim axes"""
    def __init__(self, N=1, ndim=3):
        """down sample a 2D or 3D box
        Args:
            N : int, thinning factor
            ndim : int, dimensionality of box
        """
        self.N = N
        self.ndim = ndim

    def __call__(self, box):
        if self.ndim == 2:
            return box[..., ::self.N, ::self.N]
        elif self.ndim == 3:
            return box[..., ::self.N, ::self.N, ::self.N]



