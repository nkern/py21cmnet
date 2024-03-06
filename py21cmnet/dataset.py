"""
dataset module
"""

import numpy as np
from skimage import transform
from torch.utils.data import Dataset, DataLoader
import torch
from copy import deepcopy

from . import utils


class Roll:
    """roll a box along last ndim axes"""
    def __init__(self, shift=None, ndim=3):
        """roll a periodic box by "shift" pixels
        Args:
            shift : int or tuple
                Roll the box by this many pixels
                along each of the specified dimensions.
                Default is a random number per dimension.
            ndim : int
                Dimensionality of the box
        """
        self.shift = shift
        self.ndim = ndim

    def __call__(self, box, shift=None):
        # compute shift if not fed
        if shift is None:
            if self.shift is None:
                shift = tuple(torch.randint(0, box[0].shape[-1], (self.ndim,)))
            else:
                shift = self.shift
        if isinstance(box, (list, tuple)):
            return [self.__call__(b, shift=shift) for b in box]
        if self.ndim == 2:
            return torch.roll(box, shift, dims=(-1, -2))
        elif self.ndim == 3:
            return torch.roll(box, shift, dims=(-1, -2, -3))


class DownSample:
    """down sample a box along last ndim axes"""
    def __init__(self, thin=1, ndim=3):
        """down sample a 2D or 3D box
        Args:
            thin : int
                thinning factor
            ndim : int
                dimensionality of box
        """
        self.ndim = ndim
        self.thin = thin

    def __call__(self, box):
        if isinstance(box, (list, tuple)):
            return [self.__call__(b) for b in box]
        if self.ndim == 2:
            return box[..., ::self.thin, ::self.thin]
        elif self.ndim == 3:
            return box[..., ::self.thin, ::self.thin, ::self.thin]


class Slice:
    """slice a box along last ndim axes"""
    def __init__(self, slices=None, ndim=3):
        """slice a 2D or 3D box

        Args:
            slice : slice object or tuple of slice
                axis slice. If tuple, must be len-ndim
            ndim : int
                dimensionality of box
        """
        self.ndim = ndim
        if slices is None:
            slices = [slice(None, None) for i in range(ndim)]
        if not isinstance(slices, (tuple, list)):
            slices = [slices for i in range(ndim)]
        self.slices = slices

    def __call__(self, box):
        if isinstance(box, (list, tuple)):
            return [self.__call__(b) for b in box]
        if self.ndim == 2:
            return box[..., self.slices[0], self.slices[1]]
        elif self.ndim == 3:
            return box[..., self.slices[0], self.slices[1], self.slices[2]]


class Transpose:
    """Transpose a box along last ndim axes"""
    def __init__(self, axes=None, ndim=3):
        """transpose a 2D or 3D dataset
        Args:
            axes : 0th ordered, ndim-len tuple
                This is the new axes ordering along last ndim axes.
                Default is random ordering.
                E.g. no tranpose is axes=(0, 1, 2) for a 3d box
            ndim : int
                Dimensions of box (e.g. 2d or 3d)
        """
        self.ndim = ndim
        self.axes = axes

    def __call__(self, box, axes=None):
        if isinstance(box, (list, tuple)):
            full_dim = box[0].ndim
        else:
            full_dim = box.ndim
        dim_diff = full_dim - self.ndim
        # compute axes if not fed
        if axes is None:
            if self.axes is None:
                axes = tuple(np.random.choice(range(self.ndim), self.ndim, replace=False))
            else:
                axes = self.axes
        if isinstance(box, (list, tuple)):
            return [self.__call__(b, axes=axes) for b in box]
        # modify axes for full_dim
        axes = tuple(range(dim_diff)) + tuple(np.array(axes) + dim_diff)
        return torch.permute(box, axes)


class BoxDataset(Dataset):
    """
    Dataset for cosmological box output
    """

    def __init__(self, Xfiles, yfiles, readf, transform=None,
                 X_augment=None, y_augment=None, **kwargs):
        """Cosmological box dataset

        Args:
            Xfiles : list of str, list of sublist of str, required
                List of filepaths (of len Nsamples) to box output of
                feature values. If fed as a list of sublist of str,
                each element in a sublist is a unique channel.
            yfiles : list of str, list of sublist of str, requjired
                List of filepaths to box output of target values.
                Same rules apply as Xfiles, must match len of Xfiles
            readf : callable, required
                Data read function, input as element of Xfiles or yfiles.
                If Xfiles and yfiles holds the data in-memory,
                use utils.load_dummy as readf.
            transform : callable, list of callable
                Box transformations to apply to X and y simultaneously
                for each draw, but possibly randomly between draws.
            X_augment : callable, list of callable
                Augmentation(s) to apply to Xfiles, if fed as list
                must be of len Nchannel. Feed as None for no augmentation.
            y_augment : callable, list of callable
                Augmentation(s) to apply to yfiles, if fed as list
                must be of len Nchannel. Feed as None for no augmentation.

        Notes:
            Augmentation and transformation are defined differently.
            An augmentation is an action that is independent of X or y, and
            can vary from channel to channel. To apply multiple augmentations
            to a single channel, you can compose them using dataset.ComposeAugments.
            A transformation is applied to X and y (and all channels) simultaneously.
        """
        if isinstance(Xfiles, str):
            Xfiles = [Xfiles]
        self.Xfiles = Xfiles
        if isinstance(yfiles, str):
            yfiles = [yfiles]
        self.yfiles = yfiles
        assert len(self.Xfiles) == len(self.yfiles), "Xfiles and yfiles must have same len"
        self.Nfiles = len(self.Xfiles)
        self.transform = transform
        self.readf = readf
        self.kwargs = kwargs
        self.X_augment = X_augment
        self.y_augment = y_augment

    def __len__(self):
        return len(self.Xfiles)

    def __getitem__(self, idx):
        # load box
        X = self.readf(self.Xfiles[idx], **self.kwargs)
        y = self.readf(self.yfiles[idx], **self.kwargs)

        # augment the data if requested: only makes a copy if augmenting
        X, y = self.augment(X, y, inplace=False)

        # transform the data
        if self.transform is not None:
            X, y = self.transform((X, y))

        return X, y

    def augment(self, X, y, undo=False, inplace=False):
        """Augment X and y given augmentation parameters

        Args:
            X : numpy.ndarray or torch.Tensor
                Feature data
            y : numpy.ndarray or torch.Tensor
                Target data
            undo : bool, default=False
                If True, undo the augmentation
            inplace : bool, default=False
                If True, augment data in place, otherwise
                make a deepcopy of inputs.
                ** Note: ** this only copies the data if
                an augmentation is needed. If self.X_augmentation
                is None, then X is not copied even if inplace=False.

        Returns:
            augmented X, augmented y
        """
        # augment X
        if self.X_augment is not None:
            if not inplace:
                X = deepcopy(X)
            if isinstance(self.X_augment, (list, tuple)):
                # augment each channel separately
                assert len(self.X_augment) == len(X), "X_augment len must match X"
                for i, xaug in enumerate(self.X_augment):
                    if callable(xaug):
                        # only augment if xaug is a callable
                        X[i] = xaug(X[i], undo=undo)
            else:
                if callable(self.X_augment):
                    X[:] = self.X_augment(X, undo=undo)

        # augment y
        if self.y_augment is not None:
            if not inplace:
                y = deepcopy(y)
            if isinstance(self.y_augment, (list, tuple)):
                # augment each channel separately
                assert len(self.y_augment) == len(y), "y_augment len must match y"
                for i, yaug in enumerate(self.y_augment):
                    if callable(yaug):
                        # only augment if yaug is a callable
                        y[i] = yaug(y[i], undo=undo)
            else:
                if callable(self.y_augment):
                    y[:] = self.y_augment(y, undo=undo)

        return X, y


class ComposeAugments:
    """A class to compose multiple data augmentation
    routines, similar to torchvision.transforms.Compose

    Only requirement is that the callables
    have a __call__() method that takes a single
    numpy.ndarray or torch.Tensor, and has a single
    kwarg "undo: bool False", which when set to True
    undoes the augmentation.
    """

    def __init__(self, augments):
        """Compose augmentations

        Args:
            augments : list
                List of callables to be applied
                in the order of the list
        """
        self.augments = augments

    def __call__(self, X, undo=False):
        augs = self.augments if not undo else self.augments[::-1]
        for aug in augs:
            X = aug(X, undo=undo)

        return X


class Logarithm:
    """Take logarithm of data"""

    def __init__(self, log10=True, offset=0, scale=1):
        """Take logarithm of input

        :math:`\log((x - \text{offset}) / \text{scale})`        

        Args:
            log10 : bool, default=True
                If True, take log10(), else take ln()
            offset : float, default=0
                Subtract offset to x before taking log
            scale : float, default=1
                Divide (x - offset) by scale before log

        Notes:
            To undo the action, pass the undo=True kwarg
            to the object call.
        """
        self.log10 = log10
        self.offset = offset
        self.scale = scale

    def __call__(self, box, undo=False):
        if isinstance(box, (list, tuple)):
            return [self.__call__(b, undo=undo) for b in box]
        if not undo:
            log = torch.log10 if self.log10 else torch.log
            return log((box - self.offset) / self.scale)
        else:
            func = (lambda x: 10**x) if self.log10 else torch.exp
            return func(box) * self.scale + self.offset

