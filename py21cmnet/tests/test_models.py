"""
models testing suite
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


def read_test_data(ndim=3):
    fname = os.path.join(DATA_PATH, "train_21cmfast_basic.h5")
    X = utils.load_hdf5([fname+'/deltax', fname+'/Gamma'], dtype=np.float32)
    y = utils.load_hdf5([fname+'/x_HI', fname+'/Ts'], dtype=np.float32)

    if ndim == 3:
        # cut out 8 quadrants of 3d data
        N = X.shape[-1] // 2
        _X, _y = [], []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    _X.append(X[:, i*N:(i+1)*N, j*N:(j+1)*N, k*N:(k+1)*N])
                    _y.append(y[:, i*N:(i+1)*N, j*N:(j+1)*N, k*N:(k+1)*N])
        X = np.array(_X)
        y = np.array(_y)

    elif ndim == 2:
        # cut out each slice
        _X, _y = [], []
        for i in range(X.shape[-1]):
            _X.append(X[..., i])
            _y.append(y[..., i])
        X = np.array(_X)
        y = np.array(_y)

    return torch.as_tensor(X), torch.as_tensor(y)


def test_conv():
    for ndim in [2, 3]:
        X, y = read_test_data(ndim)
        conv_kwargs = dict(in_channels=2, out_channels=5, kernel_size=3, padding=1)
        C = models.ConvNd(conv_kwargs, conv='Conv{}d'.format(ndim),
                          batch_norm='BatchNorm{}d'.format(ndim), dropout='Dropout{}d'.format(ndim))
        assert C(X).shape == X.shape[:1] + (5,) + X.shape[2:]


def test_upsample():
    for ndim in [2, 3]:
        X, y = read_test_data(ndim)
        conv_kwargs = dict(in_channels=2, out_channels=5, kernel_size=3, padding=1)
        U = models.UpSample(dict(scale_factor=2), conv_kwargs, conv='Conv{}d'.format(ndim))
        assert U(X).shape == X.shape[:1] + (5,) + tuple(np.array(X.shape[2:])*2)


def test_encoder():
    for ndim in [2, 3]:
        X, y = read_test_data(ndim)
        conv1 = dict(conv="Conv{}d".format(ndim), batch_norm='BatchNorm{}d'.format(ndim),
                     dropout='Dropout{}d'.format(ndim),
                     conv_kwargs=dict(in_channels=2, out_channels=5, kernel_size=3, padding=1))
        conv2 = dict(conv="Conv{}d".format(ndim), batch_norm='BatchNorm{}d'.format(ndim),
                     dropout='Dropout{}d'.format(ndim),
                     conv_kwargs=dict(in_channels=5, out_channels=10, kernel_size=3, padding=1))
        E = models.Encoder([conv1, conv2], maxpool='MaxPool{}d'.format(ndim), maxpool_kwargs=dict(kernel_size=2))
        assert E(X).shape == X.shape[:1] + (10,) + tuple(np.array(X.shape[2:])//2)


def test_decoder():
    for ndim in [2, 3]:
        for up_mode in ['upsample', 'ConvTranspose{}d'.format(ndim)]:
            X, y = read_test_data(ndim)
            conv1 = dict(conv="Conv{}d".format(ndim), batch_norm='BatchNorm{}d'.format(ndim),
                         dropout='Dropout{}d'.format(ndim),
                         conv_kwargs=dict(in_channels=2, out_channels=4, kernel_size=3, padding=1))
            conv2 = dict(conv="Conv{}d".format(ndim), batch_norm='BatchNorm{}d'.format(ndim),
                         dropout='Dropout{}d'.format(ndim),
                         conv_kwargs=dict(in_channels=4, out_channels=2, kernel_size=3, padding=1))
            if 'Conv' in up_mode:
                up_kwargs = dict(in_channels=2, out_channels=1, kernel_size=3, padding=1, output_padding=1, stride=2)
            else:
                up_kwargs = dict(upsample_kwargs=dict(scale_factor=2), conv='Conv{}d'.format(ndim),
                                 conv_kwargs=dict(in_channels=2, out_channels=1, kernel_size=3, padding=1))
            D = models.Decoder([conv1, conv2], up_kwargs, conv='Conv{}d'.format(ndim), up_mode=up_mode)
            assert D(X).shape == X.shape[:1] + (1,) + tuple(np.array(X.shape[2:])*2)

            # test crop and concat for skip connection
            crop = tuple(np.array(X.shape[-ndim:]) // 2)
            Xcrop = D.center_crop(X, crop)
            assert Xcrop.shape[-ndim:] == crop

            Xcrop_concat = D.crop_concat(Xcrop, X)
            assert Xcrop_concat.shape == Xcrop.shape[:1] + (Xcrop.shape[1] * 2,) + Xcrop.shape[2:]

            conv1['conv_kwargs']['in_channels'] *= 2
            D = models.Decoder([conv1, conv2], up_kwargs, conv='Conv{}d'.format(ndim), up_mode=up_mode)
            assert D(Xcrop, X).shape == X.shape[:1] + (1,) + tuple(np.array(X.shape[2:]))


def test_autoencoder(test_train=False):
    for ndim in [2, 3]:
        X, y = read_test_data(ndim)
        # load a parameter file and create an autoencoder object
        config = os.path.join(CONFIG_PATH, "autoencoder{}d.yaml".format(ndim))
        defaults = os.path.join(CONFIG_PATH, "autoencoder{}d_defaults.yaml".format(ndim))
        params = utils.load_autoencoder_params(config, defaults)
        model = models.AutoEncoder(**params)
        # pass tensor through
        out = model(X)
        assert out.shape == X.shape[:1] + (2,) + X.shape[2:]

        if test_train:
            ds = dataset.BoxDataset(X, y, utils.load_dummy, transform=dataset.Roll(ndim=ndim))
            dl = torch.utils.data.DataLoader(ds)
            info = utils.train(model, dl, torch.nn.MSELoss(reduction='mean'),
                               torch.optim.Adam, optim_kwargs=dict(lr=0.1), Nepochs=5, )

            pred = model(X[:1])
            # import matplotlib.pyplot as plt;
            # plt.plot(info['train_loss'])
            # fig,axes=plt.subplots(1,2,figsize=(10,5));axes[0].imshow(y[0,0],aspect='auto');axes[1].imshow(pred[0,0].detach().numpy(),aspect='auto')
