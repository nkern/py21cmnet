"""
dataset module
"""

import numpy as np
from skimage import transform
from torch.utils.data import Dataset, DataLoader

from . import utils


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

    def __call__(self, box, shift=None):
        # compute shift if not fed
        if shift is None:
            if self.shift is None:
                shift = np.random.randint(0, box[0].shape[-1], self.ndim)
            else:
                shift = self.shift
        if isinstance(box, (list, tuple)):
            return [self.__call__(b, shift=shift) for b in box]
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
        if isinstance(box, (list, tuple)):
            return [self.__call__(b) for b in box]
        if self.ndim == 2:
            return box[..., ::self.thin, ::self.thin]
        elif self.ndim == 3:
            return box[..., ::self.thin, ::self.thin, ::self.thin]



class BoxDataset(Dataset):
    """
    Dataset for cosmological box output
    """

    def __init__(self, Xfiles, yfiles, readf, transform=None, **kwargs):
        """Cosmological box dataset
        
        Args:
            Xfiles : list of str, list of sublist of str, required
                List of filepaths to box output of feature values
                If fed as a list of sublist of str, each element
                in a sublist is a unique channel.
            yfiles : list of str, list of sublist of str, requjired
                List of filepaths to box output of target values.
                Same rules apply as Xfiles, must match len of Xfiles
            readf : callable, required
                Data read function, input as element of Xfiles or yfiles.
                If Xfiles and yfiles holds the data in-memory,
                use utils.load_dummy as readf.
            transform : callable, list of callable
                Box transformations

        Returns:
            ndarray (Nfiles, Nchans, box_shape)
        """
        self.Xfiles = Xfiles
        self.yfiles = yfiles
        self.transform = transform
        self.readf = load_21cmfast
        self.kwargs = kwargs
        assert len(self.Xfiles) == len(self.yfiles)

    def __len__(self):
        return len(self.Xfiles)

    def __getitem__(self, idx):
        # load box
        X = self.readf(self.Xfiles[idx], **self.kwargs)
        y = self.readf(self.yfiles[idx], **self.kwargs)

        # transform
        if self.transform is not None:
            X, y = self.transform((X, y))

        return X, y
