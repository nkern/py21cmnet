"""
utility functions
"""
import numpy as np
from copy import deepcopy
import h5py
import yaml
from astropy import constants as con
from simpleqe.utils import Cosmology


def load_21cmfast(fname, dtype=np.float32, N=256):
    """
    Load 21cmfast box(es) output

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


def load_dummy(fname, copy=False):
    """Dummy load function"""
    if copy:
        return deepcopy(fname)
    else:
        return fname


def write_hdf5(box, fname, overwrite=False, params=None, **kwargs):
    """
    Write hdf5 data

    If fname is fed as *.h5/path or *.hdf5/path, will
    write as a Dataset object "path" in *.h5 or *.hdf5 file

    """


def load_hdf5(fname, dtype=np.float32, N=256):
    """
    Load hdf5 data

    """
    if isinstance(fname, (list, tuple)):
        box = np.empty((len(fname), N, N, N), dtype=dtype)
        for i, fn in enumerate(fname):
            box[i] = load_hdf5(fn, dtype=dtype, N=N)

    return box


def _update_dict(d1, d2):
    for key in d2.keys():
        if key in d1:
            if isinstance(d2[key], dict):
                _update_dict(d1[key], d2[key])
            else:
                d1[key] = d2[key]
        else:
            d1[key] = d2[key]


def _update_yaml(d):
    for k in d:
        if isinstance(d[k], dict):
            _update_yaml(d[k])
        if d[k] in ['None', 'none']:
            d[k] = None


def load_autoencoder_params(config, defaults=None):
    """
    Parse and configure py21cmnet yaml parameter file
    for the AutoEncoder model

    Args:
        config : str, required
            Path to primary yaml param file
            E.g. py21cmnet/config/autoencoder2d.yaml
        defaults : str, optional
            Path to ancillary yaml, holding defaults
            E.g. py21cmnet/config/autoencoder2d_defaults.yaml

    Returns: arguments of model.AutoEncoder()
        encoder_layers : dict
            list of encoder blocks
        decoder_layers : dict
            list of decoder blocks
        final_layer : dict
            final layer block
        connections : dict
            skip connection mapping
        final_transforms : list
            final transformations on output
    """
    # open files
    with open(config, 'r') as f:
        p = yaml.load(f, Loader=yaml.FullLoader)
        _update_yaml(p)
    if defaults is not None:
        with open(defaults, 'r') as f:
            d = yaml.load(f, Loader=yaml.FullLoader)
            _update_yaml(d)
    else:
        d = {}

    # sort encoder block
    encoder_layers = []
    encoders = sorted([k for k in p.keys() if 'encode' in k])  # careful w/ sorting here
    # iterate over each encoder block
    for block in encoders:
        # get default parameters
        encode = deepcopy(d.get('encode', {}))
        # iterate over whats in primary and overwrite
        for param in p[block]:
            encode[param] = p[block][param]
            # special handling for conv_layers
            if param == 'conv_layers' and 'conv_layer' in d:
                # iterate over each conv layer in the primary
                for i, layer in enumerate(encode['conv_layers']):
                    # update defaults with what's in primary
                    conv_layer = deepcopy(d['conv_layer'])
                    _update_dict(conv_layer, layer)
                    encode['conv_layers'][i] = conv_layer
        encoder_layers.append(encode)

    # sort decoder block
    decoder_layers = []
    decoders = sorted([k for k in p.keys() if 'decode' in k])  # careful w/ sorting here
    # iterate over each decoder block
    for block in decoders:
        # get default parameters
        decode = deepcopy(d.get('decode', {}))
        # iterate over whats in primary and overwrite
        for param in p[block]:
            decode[param] = p[block][param]
            # special handling for conv_layers
            if param == 'conv_layers' and 'conv_layer' in d:
                # iterate over each conv layer in the primary
                for i, layer in enumerate(decode['conv_layers']):
                    # update defaults with what's in primary
                    conv_layer = deepcopy(d['conv_layer'])
                    _update_dict(conv_layer, layer)
                    decode['conv_layers'][i] = conv_layer
            # special handling for up_kwargs
            elif param == 'up_kwargs' and 'up_kwargs' in d['decode']:
                up_kwargs = deepcopy(d['decode']['up_kwargs'])
                _update_dict(up_kwargs, decode['up_kwargs'])
                decode['up_kwargs'] = up_kwargs

        decoder_layers.append(decode)

    # sort final layer
    final_layer = p['final']
    if 'conv_layer' in d:
        for i, layer in enumerate(final_layer['conv_layers']):
            # update defaults with what's in primary
            conv_layer = deepcopy(d['conv_layer'])
            _update_dict(conv_layer, layer)
            final_layer['conv_layers'][i] = conv_layer


    network = dict(encoder_layers=encoder_layers, decoder_layers=decoder_layers,
                   final_layer=final_layer, connections=p['connections'],
                   final_transforms=p['final_transforms'])

    return network



