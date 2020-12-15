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

    # roll the cube
    Roll = dataset.Roll(50, ndim=3)
    assert Roll(db).shape == db.shape
    assert not (Roll(db) == db).any()

    # roll the box
    assert Roll(box).shape == box.shape
    assert not (Roll(box) == box).any()

    # downsample
    DS = dataset.DownSample(2, ndim=3)
    assert DS(db).shape == tuple(np.array(db.shape)/2)
    assert (DS(db) == db[::2, ::2, ::2]).all()
    assert DS(box).shape == box.shape[:1] + tuple(np.array(box.shape[1:])/2)

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


def test_dataset():
    fname = os.path.join(DATA_PATH, "train_21cmfast_basic.h5")
    Xfiles = [[fname+'/deltax', fname+'/Gamma']]
    yfiles = [[fname+'/x_HI', fname+'/Ts']]
    dtype = np.float32

    # simple load
    X = utils.load_hdf5(Xfiles[0], dtype=dtype)
    y = utils.load_hdf5(yfiles[0], dtype=dtype)
    dl = dataset.BoxDataset(Xfiles, yfiles, utils.load_hdf5, dtype=dtype)
    assert len(dl) == 1
    assert (dl[0][0] == X).all()
    assert (dl[0][1] == y).all()

    # load with dummy
    dl = dataset.BoxDataset([X], [y], utils.load_dummy)
    assert len(dl) == 1
    assert (dl[0][0] == X).all()
    assert (dl[0][1] == y).all()

    # load with transformation
    trans = Compose([dataset.Roll(shift=20, ndim=3), dataset.DownSample(thin=2, ndim=3)])
    dl = dataset.BoxDataset(Xfiles, yfiles, utils.load_hdf5, dtype=dtype, transform=trans)
    assert len(dl) == 1
    assert dl[0][0].shape == X.shape[:1] + tuple(np.array(X.shape[1:])/2)
    assert not (dl[0][0][0] == X[0, ::2, ::2, ::2]).any()




