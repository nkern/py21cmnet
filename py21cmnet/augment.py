"""
data augmentation module
"""

import numpy as np
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
        if box is None:
            return box

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
        if box is None:
            return box

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
        if box is None:
            return box

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
        if box is None:
            return box

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


class Rot90:
    """
    Rotate ND images by 90 degrees
    """
    def __init__(self, k=None, dims=(0, 1)):
        """
        Parameters
        ----------
        k : int, optional
            Number of times to rotate. Default
            is a random number of times [0, 3]
        dims : tuple, optional
            Dimensions to rotate
        """
        self.k = k
        self.dims = dims

    def __call__(self, box, k=None, dims=None):
        if box is None:
            return box

        k = k if k is not None else self.k
        dims = dims if dims is not None else self.dims

        if k is None:
            k = torch.randint(0, 4, (1,))[0]

        if isinstance(box, (list, tuple)):
            return [self.__call__(b, k=k, dims=dims) for b in box]

        return torch.rot90(box, k=k, dims=dims)


class Crop:
    """
    Crop an Nd image / cube.
    """
    def __init__(self, size, low, high):
        """
        Parameters
        ----------
        size : tuple of int
            Size of last N dimensions after crop.
        low : tuple of int
            N-dim tuple holding lowest index
            for crop along last N dims.
        high : tuple of int
            N-dim tuple holding highest index
            for crop along last N dims.
        """
        self.size = size
        self.low = low
        self.high = high

    def __call__(self, box, crop=None):
        if box is None:
            return box

        if crop is None:
            crop = []
            for s, l, h in zip(self.size, self.low, self.high):
                i = torch.randint(l, h, (1,))
                crop.append(slice(i, i + s))
            crop = tuple(crop)

        if isinstance(box, (list, tuple)):
            return [self.__call__(b, crop=crop) for b in box]

        return box[(...,) + tuple(crop)]


class RectMask:
    """
    Rectangular masking
    """
    def __init__(self, size, low, high, N=1, store_mask=None, inplace=False):
        """
        Parameters
        ----------
        size : tuple of int
            Size of last N dimensions of mask.
        low : tuple of int
            N-dim tuple holding lowest index
            for mask along last N dims.
        high : tuple of int
            N-dim tuple holding highest index
            for mask along last N dims.
        N : int
            Number of masked regions to make
        store_mask : dict
            Store the mask here as 'mask'
        """
        self.size = size
        self.low = low
        self.high = high
        self.N = N
        self.store_mask = store_mask
        self.inplace = inplace

    def __call__(self, box, mask=None, inplace=None, **kwargs):
        if box is None:
            return box

        if mask is None:
            mask = torch.ones_like(box)
            for i in range(self.N):
                crop = []
                for s, l, h in zip(self.size, self.low, self.high):
                    start = torch.randint(l, h, (1,))
                    crop.append(slice(start, start + s))
                mask[..., *tuple(crop)] = 0.0

        if isinstance(box, (list, tuple)):
            return [self.__call__(b, mask=mask) for b in box]

        if self.store_mask is not None:
            self.store_mask['mask'] = mask

        inplace = inplace if inplace is not None else self.inplace
        if inplace:
            box *= mask
            return box

        else:
            return box * mask


class MaskWeight:
    """
    Construct loss function weights given
    image mask. Feed self.store_mask to 
    input of RectMask or other masking augmentations.
    """
    def __init__(self):
        self.store_mask = {}

    def __call__(self, *args, **kwargs):
        w = None
        if 'mask' in self.store_mask:
            w = 1 - self.store_mask['mask']

        return w


class GaussNoise:
    """
    Add gaussian noise
    """
    def __init__(self, shape, amp=1.0, device=None, dtype=None, inplace=False):
        self.shape = shape
        self.device = device
        self.amp = amp
        self.dtype = dtype
        self.inplace = inplace

    def __call__(self, box, noise=None):
        if box is None:
            return box

        if noise is None:
            noise = torch.randn(shape, device=self.device, dtype=self.dtype) * self.amp

        if isinstance(box, (list, tuple)):
            return [self.__call__(b, noise=noise) for b in box]

        if self.inplace:
            box += noise
        else:
            box = box + noise

        return box


class ComposeTransforms:
    """
    A class to compose multiple transformations.
    """
    def __init__(self, transforms):
        """
        Parameters
        ----------
        transforms : list
            List of callables to be applied
            in the order of the list
        """
        self.transforms = transforms

    def __call__(self, X, **kwargs):
        for transform in self.transforms:
            X = transform(X)

        return X


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

    def __call__(self, X, undo=False, **kwargs):
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
        if box is None:
            return box

        if isinstance(box, (list, tuple)):
            return [self.__call__(b, undo=undo) for b in box]
        if not undo:
            log = torch.log10 if self.log10 else torch.log
            return log((box - self.offset) / self.scale)
        else:
            func = (lambda x: 10**x) if self.log10 else torch.exp
            return func(box) * self.scale + self.offset
