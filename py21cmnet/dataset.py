"""
dataset module
"""
import numpy as np
from skimage import transform
from torch.utils.data import Dataset, DataLoader

from . import utils


def load_21cmfast(fname, dtype=np.float32, N=512):
    """load 21cmfast box(es)
    Args:
        fname : str or list of str
            If list of str, prepend boxes
        dtype : datatype
        N : int
            size of box length in pixels
    """
    if isinstance(fname, (list, tuple)):
        box = np.empty((len(fname), N, N, N), dtype=dtype)
        for i, fn in enumerate(fname):
            box[i] = load_21cmfast(fn, dtype=dtype, N=N)
    else:
        box = np.fromfile(fname, dtype=dtype).reshape(N, N, N)

    return box


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
        if self.ndim == 2:
            return np.roll(box, shift, axis=(-1, -2))
        elif self.ndim == 3:
            return np.roll(box, shift, axis=(-1, -2, -3))


class DownSample:
    """down sample a box along last ndim axes"""
    def __init__(self, thin=1, ndim=3):
        """down sample a 2D or 3D box
        Args:
            thin : int, thinning factor
            ndim : int, dimensionality of box
        """
        self.thin = thin
        self.ndim = ndim

    def __call__(self, box):
        if self.ndim == 2:
            return box[..., ::self.thin, ::self.thin]
        elif self.ndim == 3:
            return box[..., ::self.thin, ::self.thin, ::self.thin]



class BoxDataset(Dataset):

    def __init__(self, bfiles, transform=None, readf=load_21cmfast, **kwargs):
        """Cosmological box dataset
        
        Args:
            bfiles : list of str, or list of sublist of str
                List of filepaths to box output.
                If fed as a list of sublist of str, each element
                in a sublist is a unique channel.
            transform : callable, list of callable
                Box transformations
            readf : callable
                box read function

        Notes:
            Returns ndarray (Nfiles, Nchans, box_shape)
        """
        if isinstance(bfiles, str):
            bfiles = [bfiles]
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











