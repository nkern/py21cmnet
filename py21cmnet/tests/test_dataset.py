"""
dataset testing suite
"""
import numpy as np
import torch
from torchvision.transforms import Compose
import os
import yaml

from py21cmnet import utils, models, dataset
from py21cmnet.config import CONFIG_PATH
from py21cmnet.data import DATA_PATH

import pytest


def test_transforms():
    fname = os.path.join(DATA_PATH, "train_21cmfast_basic.h5")
    db = utils.load_hdf5(fname + '/deltax', dtype=np.float32)
    box = utils.load_hdf5([fname + '/deltax', fname + '/Ts'], dtype=np.float32)

    db, box = torch.as_tensor(db), torch.as_tensor(box)

    # roll the cube
    Roll = dataset.Roll((50, 50, 50), ndim=3)
    assert Roll(db).shape == db.shape
    assert not (Roll(db) == db).any()

    # roll the box
    assert Roll(box).shape == box.shape
    assert not (Roll(box) == box).any()

    # downsample
    DS = dataset.DownSample(2, ndim=3)
    assert DS(db).shape == torch.Size(np.array(db.shape)//2)
    assert (DS(db) == db[::2, ::2, ::2]).all()
    assert DS(box).shape == box.shape[:1] + torch.Size(np.array(box.shape[1:])//2)

    # transpose
    db_mod = db[:, ::2, ::4]
    DS = dataset.Transpose(ndim=3)
    # test fed axes
    db_tran = DS(db_mod, axes=(1, 2, 0))
    assert db_tran.shape == (64, 32, 128)
    assert np.isclose(db_tran, np.transpose(db_mod, (1, 2, 0))).all()
    # test random axes
    np.random.seed(1)
    db_tran = DS(db_mod)
    assert db_tran.shape == (128, 32, 64)
    # test box
    box_mod = box[:, :, ::2, ::4]
    box_tran = DS(box_mod, axes=(1, 2, 0))
    assert box_tran.shape == (2, 64, 32, 128)
    # test list
    box_tran = DS([box_mod[0], box_mod[1]], axes=(1, 2, 0))
    for i in range(2):
        assert box_tran[i].shape == (64, 32, 128)

    # slice
    S = dataset.Slice(slices=[slice(0, 10), slice(None), slice(None)])
    db_mod = S(db)
    assert db_mod.shape == (10, 128, 128)
    S = dataset.Slice(slices=slice(0, 10))
    db_mod = S(db)
    assert db_mod.shape == (10, 10, 10)
    # test box
    box_mod = S(box)
    assert box_mod.shape == (2, 10, 10, 10)
    # test list
    box_mod = S([box[0], box[1]])
    for i in range(2):
        assert box_mod[i].shape == (10, 10, 10)

def test_dataset():
    fname = os.path.join(DATA_PATH, "train_21cmfast_basic.h5")
    Xfiles = [[fname+'/deltax', fname+'/Gamma']]
    yfiles = [[fname+'/x_HI', fname+'/Ts']]
    dtype = np.float32

    # simple load
    X = utils.load_hdf5_torch(Xfiles[0], dtype=dtype)
    y = utils.load_hdf5_torch(yfiles[0], dtype=dtype)
    dl = dataset.BoxDataset(Xfiles, yfiles, utils.load_hdf5_torch, dtype=dtype)
    assert len(dl) == 1
    assert (dl[0][0] == X).all()
    assert (dl[0][1] == y).all()

    # load with dummy
    dl = dataset.BoxDataset([X], [y], utils.load_dummy)
    assert len(dl) == 1
    assert (dl[0][0] == X).all()
    assert (dl[0][1] == y).all()

    # load with transformation
    trans = Compose([dataset.Roll(shift=(20,20,20), ndim=3), dataset.DownSample(thin=2, ndim=3)])
    dl = dataset.BoxDataset(Xfiles, yfiles, utils.load_hdf5_torch, dtype=dtype, transform=trans)
    assert len(dl) == 1
    assert dl[0][0].shape == X.shape[:1] + torch.Size(np.array(X.shape[1:])//2)
    assert not (dl[0][0][0] == X[0, ::2, ::2, ::2]).any()

def test_augmentations():
    fname = os.path.join(DATA_PATH, "train_21cmfast_basic.h5")
    Xfiles = [[fname+'/deltax', fname+'/Gamma']]
    yfiles = [[fname+'/x_HI', fname+'/Ts']]
    dtype = np.float32
    X = utils.load_hdf5_torch(Xfiles[0], dtype=dtype)
    y = utils.load_hdf5_torch(yfiles[0], dtype=dtype)

    # test single augmentation
    aug = dataset.Logarithm(offset=-1)
    X_aug = aug(X[0])
    assert not np.isnan(X_aug).any()
    assert np.isclose(X[0], aug(X_aug, undo=True), atol=1e-6).all()

    # test composed augmentation
    def shift(x, undo=False):
        if not undo:
            return x + 5
        else:
            return x - 5

    multi_aug = dataset.ComposeAugments([aug, shift])
    X_aug2 = multi_aug(X[0])
    assert np.isclose(X_aug + 5, X_aug2, atol=1e-7).all()
    assert np.isclose(X[0], multi_aug(X_aug2, undo=True), atol=1e-6).all()

    # try with dataset: only one augmentation for both X and y
    Xaugment, yaugment = dataset.Logarithm(offset=-1), dataset.Logarithm(offset=-1)
    dl = dataset.BoxDataset(Xfiles, yfiles, utils.load_hdf5_torch, dtype=dtype)
    dl_aug = dataset.BoxDataset(Xfiles, yfiles, utils.load_hdf5_torch, dtype=dtype,
                                X_augment=Xaugment, y_augment=yaugment)
    X, y = dl[0]
    Xaug, yaug = dl_aug[0]
    # check they were augmented
    assert np.isclose(Xaug, Xaugment(X), atol=1e-6).all()
    assert np.isclose(yaug, yaugment(y), atol=1e-6).all()
    # check undo
    assert np.isclose(X, dl_aug.augment(Xaug, yaug, undo=True)[0], atol=1e-6).all()
    assert np.isclose(y, dl_aug.augment(Xaug, yaug, undo=True)[1], atol=1e-6).all()

    # try with some augmentation for X and y channels
    Xaugment = [dataset.Logarithm(offset=-1), None]
    yaugment = [None, dataset.Logarithm()]
    dl = dataset.BoxDataset(Xfiles, yfiles, utils.load_hdf5_torch, dtype=dtype)
    dl_aug = dataset.BoxDataset(Xfiles, yfiles, utils.load_hdf5_torch, dtype=dtype,
                                X_augment=Xaugment, y_augment=yaugment)
    X, y = dl[0]
    Xaug, yaug = dl_aug[0]
    # check zeroth channe augmented but not first (vice versa for y)
    assert np.isclose(Xaug[0], Xaugment[0](X[0]), atol=1e-6).all()
    assert np.isclose(Xaug[1], X[1], atol=1e-6).any()
    assert np.isclose(yaug[1], yaugment[1](y[1]), atol=1e-6).all()
    assert np.isclose(yaug[0], y[0], atol=1e-6).any()
    # check undo
    assert np.isclose(X, dl_aug.augment(Xaug, yaug, undo=True)[0], atol=1e-6).all()
    assert np.isclose(y, dl_aug.augment(Xaug, yaug, undo=True)[1], atol=1e-6).all()

    # try with no aug for y and check memory location
    _X = utils.load_hdf5_torch(Xfiles[0], dtype=dtype)
    _y = utils.load_hdf5_torch(yfiles[0], dtype=dtype)
    Xaugment, yaugment = dataset.Logarithm(offset=-1), None
    dl_aug = dataset.BoxDataset([_X], [_y], utils.load_dummy,
                                X_augment=Xaugment, y_augment=yaugment)
    Xaug, yaug = dl_aug[0]
    # assert X was copied, but y was not, even though inplace=False in augmentation
    assert hex(id(_X)) != hex(id(Xaug))
    assert hex(id(_y)) == hex(id(yaug))

    # check inplace augmentation
    _X = utils.load_hdf5_torch(Xfiles[0], dtype=dtype)
    _y = utils.load_hdf5_torch(yfiles[0], dtype=dtype)
    Xaugment, yaugment = dataset.Logarithm(offset=-1), dataset.Logarithm(offset=-1)
    dl_aug = dataset.BoxDataset([_X], [_y], utils.load_dummy,
                                X_augment=Xaugment, y_augment=yaugment)
    Xaug, yaug = dl_aug.augment(_X, _y, inplace=True)
    # check that it did indeed augment
    assert not np.isclose(_X, utils.load_hdf5_torch(Xfiles[0], dtype=dtype), atol=1e-6).all()
    assert not np.isclose(_y, utils.load_hdf5_torch(yfiles[0], dtype=dtype), atol=1e-6).all()
    # check memory address is the same
    assert hex(id(_X)) == hex(id(Xaug))
    assert hex(id(_y)) == hex(id(yaug))
    # check reverse inplace augmentation
    dl_aug.augment(_X, _y, undo=True, inplace=True)
    assert np.isclose(_X, utils.load_hdf5_torch(Xfiles[0], dtype=dtype), atol=1e-6).all()
    assert np.isclose(_y, utils.load_hdf5_torch(yfiles[0], dtype=dtype), atol=1e-6).all()
