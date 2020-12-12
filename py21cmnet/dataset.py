"""
dataset module
"""
import numpy as np
from skimage import transform
from torch.utils.data import Dataset, DataLoader

from . import utils


def load_21cmfast(fname, dtype=np.float32, N=512):
    """load 21cmfast box
    Args:
        fname : str
        dtype : datatype
        N : int, size of box length in pixels
    """
    return np.fromfile(fname, dtype=dtype).reshape(N, N, N)


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



class BoxDataset(Dataset):

    def __init__(self, bfiles, transform=None, readf=load_21cmfast, **kwargs):
        """Cosmological box dataset
        
        Args:
            bfiles : str, list of str
                List of filepaths to box output
            transform : callable, list of callable
                Box transformations
            readf : callable
                box read function

        """
        self.bfiles = bfiles
        self.transform = transform
        self.readf = load_21cmfast
        self.kwargs = kwargs

    def __len__(self):
        return len(self.bfiles)

    def __getitem__(self, idx):
        # load box
        box = self.readf(self.bfiles[idx], **self.kwargs)

        # transform
        if self.transform is not None:
            box = self.transform(box)

        return box











