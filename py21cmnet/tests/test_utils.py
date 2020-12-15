"""
utils testing suite
"""
import numpy as np
import torch
import os
import yaml

from py21cmnet import utils, models
from py21cmnet.config import CONFIG_PATH
from py21cmnet.data import DATA_PATH

import pytest


def test_load_paramfile():
    for ndim in [2, 3]:
        # load a parameter file and create an autoencoder object
        config = os.path.join(CONFIG_PATH, "autoencoder{}d.yaml".format(ndim))
        with open(config, 'r') as f:
            config_p = yaml.load(f, Loader=yaml.FullLoader)
        defaults = os.path.join(CONFIG_PATH, "autoencoder{}d_defaults.yaml".format(ndim))
        with open(defaults, 'r') as f:
            defaults_p = yaml.load(f, Loader=yaml.FullLoader)

        # load config with defaults
        params = utils.load_autoencoder_params(config, defaults)

        # assert all necessary elements are present
        assert sorted(params.keys()) == ['connections', 'decoder_layers', 'encoder_layers', 'final_layer', 'final_transforms']

        # assert defaults and primaries exist
        for i, encoder in enumerate(params['encoder_layers']):
            assert encoder['maxpool'] == defaults_p['encode']['maxpool']
            for j, layer in enumerate(encoder['conv_layers']):
                assert layer['conv'] == defaults_p['conv_layer']['conv']
                assert layer['conv_kwargs']['padding'] == defaults_p['conv_layer']['conv_kwargs']['padding']
                assert layer['conv_kwargs']['in_channels'] == config_p['encode{}'.format(i)]['conv_layers'][j]['conv_kwargs']['in_channels']

        # assert defaults and primaries exist
        for i, decoder in enumerate(params['decoder_layers']):
            assert decoder['up_mode'] == defaults_p['decode']['up_mode']
            for j, layer in enumerate(decoder['conv_layers']):
                assert layer['conv'] == defaults_p['conv_layer']['conv']
                assert layer['conv_kwargs']['padding'] == defaults_p['conv_layer']['conv_kwargs']['padding']
                assert layer['conv_kwargs']['in_channels'] == config_p['decode{}'.format(i)]['conv_layers'][j]['conv_kwargs']['in_channels']

        # assert default overwritten by primary when present
        assert params['final_layer']['conv_layers'][-1]['conv_kwargs']['padding'] == config_p['final']['conv_layers'][-1]['conv_kwargs']['padding']

        # try instantiating an AutoEncoder object
        model = models.AutoEncoder(**params)

        # try running a tensor through it
        X = torch.randn((2, 2) + tuple([64] * ndim))
        out = model(X)
        assert out.shape == (2, 2) + tuple([64] * ndim)


def test_load_21cmfast():
    a = np.random.randn(1000).astype(np.float).reshape(10, 10, 10)
    fname = os.path.join(DATA_PATH, "_xyz_testfile")
    a.tofile(fname)

    box = utils.load_21cmfast(fname, dtype=np.float, N=10)
    assert np.isclose(a, box).all()

    box = utils.load_21cmfast([fname, fname], dtype=np.float, N=10)
    assert np.isclose(a, np.array([box, box])).all()

    os.remove(fname)


def test_read_write_hdf5():
    fname = os.path.join(DATA_PATH, "train_21cmfast_basic.h5")

    # ensure fname split
    fn_split = utils._split_hdf5(fname + '/deltax')
    assert fn_split[0] == fname
    assert fn_split[1] == 'deltax'

    # load single file
    db = utils.load_hdf5(fname + '/deltax')
    assert db.shape == (128, 128, 128)
    db = utils.load_hdf5(fname + '/deltax', dtype=np.float32)
    assert db.dtype == np.float32
 
    # load multi-file
    box = utils.load_hdf5([fname + '/deltax', fname + '/Gamma'], dtype=np.float32)
    assert box.shape == (2, 128, 128, 128)
    assert box.dtype == np.float32

    # write
    oname = os.path.join(DATA_PATH, "_xyz_test_train.h5")
    utils.write_hdf5(db, oname + "/deltax", overwrite=True, dtype=np.float32, params='the history', verbose=False)
    assert os.path.exists(oname)
    utils.write_hdf5(db, oname + "/deltax", overwrite=True, dtype=np.float64, params='the history', verbose=False)
    db_load = utils.load_hdf5(oname + "/deltax")
    assert db_load.dtype == np.float64
    assert np.isclose(db_load, db).all()
    os.remove(oname)

    # exceptions
    pytest.raises(ValueError, utils.load_hdf5, fname)

