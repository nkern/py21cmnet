"""
utils testing suite
"""
import numpy as np
import torch
import os
import yaml

from py21cmnet import utils, models
from py21cmnet.config import CONFIG_PATH


def test_load_paramfile():
    # load a parameter file and create an autoencoder object
    config = os.path.join(CONFIG_PATH, "autoencoder2d.yaml")
    with open(config, 'r') as f:
        config_p = yaml.load(f, Loader=yaml.FullLoader)
    defaults = os.path.join(CONFIG_PATH, "autoencoder2d_defaults.yaml")
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
            assert layer['conv_kwargs']['in_channels'] == config_p['encode{}'.format(i+1)]['conv_layers'][j]['conv_kwargs']['in_channels']

    # assert defaults and primaries exist
    for i, decoder in enumerate(params['decoder_layers']):
        assert decoder['up_mode'] == defaults_p['decode']['up_mode']
        for j, layer in enumerate(decoder['conv_layers']):
            assert layer['conv'] == defaults_p['conv_layer']['conv']
            assert layer['conv_kwargs']['padding'] == defaults_p['conv_layer']['conv_kwargs']['padding']
            assert layer['conv_kwargs']['in_channels'] == config_p['decode{}'.format(i+1)]['conv_layers'][j]['conv_kwargs']['in_channels']

    # assert default overwritten by primary when present
    assert params['final_layer']['conv_layers'][-1]['conv_kwargs']['padding'] == config_p['final']['conv_layers'][-1]['conv_kwargs']['padding']

    # try instantiating an AutoEncoder object
    model = models.AutoEncoder(**params)

    # try running a tensor through it
    X = torch.randn((2, 2, 64, 64))
    out = model(X)
    assert out.shape == (2, 1, 64, 64)
