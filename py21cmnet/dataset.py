"""
dataset module
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from copy import deepcopy

from . import utils, augment

 
class BoxDataset(Dataset):
    """
    Dataset for cosmological box output
    """
    def __init__(self, Xfiles, yfiles, read_X=None, read_y=None, get_w=None,
                 transform=None, X_augment=None, y_augment=None):
        """
        Cosmological box dataset

        Parameters
        ----------
        Xfiles : list of str, list of sublist of str, required
            List of filepaths (of len Nsamples) to box output of
            feature values. If fed as a list of sublist of str,
            each element in a sublist is a unique channel.
        yfiles : list of str, list of sublist of str, requjired
            List of filepaths to box output of target values.
            Same rules apply as Xfiles, must match len of Xfiles
        read_X : callable, optional
            Data read function for X, input as element of Xfiles.
            Default assumes Xfiles hold tensors in memory,
            so read_X is utils.LoadDummy.
        read_y : callable, optional
            Data read function for y, input as element of yfiles.
            Default assumes yfiles hold tensors in memory,
            so read_y is utils.LoadDummy.
        get_w : callable, optional
            Construct the loss function weights of the forward pass.
            Takes transformed & augmented (X, y) and returns weights
            with shape matching targets, y.
        transform : callable, list of callable
            Box transformations to apply to X and y simultaneously
            for each draw, but possibly randomly between draws.
        X_augment : callable, list of callable
            Augmentation(s) to apply to Xfiles, if fed as list
            must be of len Nchannel. Feed as None for no augmentation.
        y_augment : callable, list of callable
            Augmentation(s) to apply to yfiles, if fed as list
            must be of len Nchannel. Feed as None for no augmentation.

        Notes
        -----
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
        if read_X is None:
            read_X = utils.LoadDummy()
        if read_y is None:
            read_y = utils.LoadDummy()
        self.read_X = read_X
        self.read_y = read_y
        self.X_augment = X_augment
        self.y_augment = y_augment
        self.get_w = get_w

    def __len__(self):
        return len(self.Xfiles)

    def __getitem__(self, idx):
        # load box
        X = self.read_X(self.Xfiles[idx])
        y = self.read_y(self.yfiles[idx])

        # transform the data
        if self.transform is not None:
            X, y = self.transform((X, y))

        # augment the data if requested
        X, y = self.augment(X, y)

        # get weights
        if self.get_w is not None:
            w = self.get_w(X, y)

            return X, y, w

        return X, y

    def augment(self, X, y, undo=False):
        """Augment X and y given augmentation parameters

        Parameters
        ----------
        X : numpy.ndarray or torch.Tensor
            Feature data
        y : numpy.ndarray or torch.Tensor
            Target data
        undo : bool, default=False
            If True, undo the augmentation

        Returns
        -------
        augmented X, augmented y
        """
        # augment X
        if self.X_augment is not None:
            if isinstance(self.X_augment, (list, tuple)):
                # augment each channel separately
                assert len(self.X_augment) == len(X), "X_augment len must match X"
                X = deepcopy(X)
                for i, xaug in enumerate(self.X_augment):
                    if callable(xaug):
                        # only augment if xaug is a callable
                        X[i] = xaug(X[i], undo=undo)
            else:
                if callable(self.X_augment):
                    X = self.X_augment(X, undo=undo)

        # augment y
        if self.y_augment is not None:
            if isinstance(self.y_augment, (list, tuple)):
                # augment each channel separately
                assert len(self.y_augment) == len(y), "y_augment len must match y"
                y = deepcopy(y)
                for i, yaug in enumerate(self.y_augment):
                    if callable(yaug):
                        # only augment if yaug is a callable
                        y[i] = yaug(y[i], undo=undo)
            else:
                if callable(self.y_augment):
                    y = self.y_augment(y, undo=undo)

        return X, y


