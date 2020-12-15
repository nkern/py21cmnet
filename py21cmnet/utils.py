"""
utility functions
"""
import numpy as np
from copy import deepcopy
import h5py
import yaml
from astropy import constants as con
from simpleqe.utils import Cosmology
import os
import torch
import time
from IPython.display import clear_output


def train(model, train_dloader, loss_fn, optim, optim_kwargs={},
          acc_fn=None, Nepochs=1, valid_dloader=None, cuda=False):
    """
    Model training function

    Args:
        model : torch.nn.Module object
            A torch model to train.
            Wrap with DataParallel for multi-GPU training.
        train_dloader : DataLoader object
            A DataLoader wrapped around a Dataset object, which returns X, y
            where X is the mini-batch feature tensor, and y is the labels.
        loss_fn : callable
            Loss function. Should have a 'mean' reduction.
        optim : callable
            Optimizer function
        optim_kwargs : dict, default = {}
            Optimizer function keyword arguments
        acc_fn : callable, default = None
            Accuracy function
        Nepochs : int, default = 1
            Number of training epochs
        valid_dloader : DataLoader object, default = None
            Similar to train_dloader, but for a validation dataset
        cuda : bool, default=False
            For running on GPUs, set to True

    Returns:
        dict
            Dictionary with training info
    """
    start = time.time()
    if cuda:
        model.cuda()

    # setup optimizer
    optimizer = optim(model.parameters(), **optim_kwargs)

    # setup
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # iterate over epochs
    for epoch in range(Nepochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # training and validation
        for phase in ['train', 'valid']:
            if phase == 'train':
                # training phase
                model.train(True)
                dataloader = train_dloader
            elif valid_dloader is not None:
                # validation phase
                model.train(False)
                dataloader = valid_dloader
            else:
                # no validation phase
                continue

            running_loss = 0.0
            step = 1  # this should start at 1, not 0
            optimizer.zero_grad()

            # iterate over data
            for i, (X, y) in enumerate(dataloader):
                if cuda:
                    X = X.cuda()
                    y = y.cuda()

                # forward pass
                if phase == 'train':
                    # compute model and loss
                    out = model(X)
                    loss = loss_fn(out, y)

                    # backprop
                    loss.backward()

                    # step
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    with torch.no_grad():
                        # compute model and loss
                        out = model(x)
                        loss = loss_fn(out, y)

                # compute accuracy
                if acc_fn is not None:
                    acc = acc_fn(out, y)
                else:
                    acc = 0

                running_acc  += acc * X.shape[0]
                running_loss += loss * X.shape[0]

                if i % 10 == 0:
                    print('Current step: {}  Loss: {}'.format(i, loss))
                    if cuda:
                        print("AllocMem (Mb) {}".format(torch.cuda.memory_allocated()/1024/1024))
                        print(torch.cuda.memory_summary())

                step += 1

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            clear_output(wait=True)
            print('Epoch {}/{}'.format(epoch, Nepochs))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    info = dict(train_loss=train_loss, valid_loss=valid_loss, train_acc=train_acc, valid_acc=valid_acc,
                optimizer=optimizer)

    return info


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


def _split_hdf5(fname):
    """split hdf5 filename of .h5 or .hdf5 prefix and suffix
    E.g. /users/nkern/myfile.h5/dataset_name
        -> /users/nkern/myfile.h5, dataset_name
    E.g. /users/nkern/myfile.h5
        -> /users/nkern/myfile.h5, None
    """
    split = None
    for suff in ['.h5', '.hdf5']:
        if suff in fname:
            if len(fname.split(suff)) > 2:
                raise ValueError("Multiple appearances of {} in {}".format(suff, fname))
            split = fname.split(suff)
            split[0] += suff
            if split[1] == '':
                split[1] = None
            else:
                split[1] = split[1][1:]
            break

    return split


def write_hdf5(data, fname, overwrite=False, params=None, dtype=None, verbose=True, **kwargs):
    """
    Write hdf5 data

    If fname is fed as *.h5/path or *.hdf5/path, will
    write as a Dataset object "path" in *.h5 or *.hdf5 file

    Args:
        data : ndarray, required
            data to write to file
        fname : str, required
            filepath to write. can feed as *.h5/dataset_name
        overwrite : bool, default=False
            If True and file and/or dataset exists, overwrite it
        params : str, default=None
            File history to append to params attribute
        dtype : dtype object
            Datatype of data to overwrite, default is ndarray dtype
        kwargs : dict
            Additional kwargs to f.create_dataset
        verbose : str, default=True
            Print feedback to stdout
    """
    # split fname
    fname = _split_hdf5(fname)
    if os.path.exists(fname[0]) and fname[1] is None and not overwrite:
        if verbose: print("{} exists, overwrite is False".format(fname[0]))
        return None

    mode = 'r+' if os.path.exists(fname[0]) else 'w'
    with h5py.File(fname[0], mode) as f:
        # get current datasets
        dsets = list(f.keys())
        if fname[1] is None:
            indx = 1
            while True:
                dset_name = 'dataset{:d}'.format(indx)
                if dset_name in dsets:
                    continue
                break
        else:
            dset_name = fname[1]
        # create dataset
        if dtype is None:
            dtype = data.dtype
        if dset_name in dsets and not overwrite:
            if verbose: print("{} exists, overwrite is False".format(fname))
            return None
        elif dset_name in dsets and overwrite:
            del f[dset_name]
        if verbose: print("writing {}/{}".format(fname[0], dset_name))
        f.create_dataset(dset_name, data=data, dtype=dtype, **kwargs)

        # update params
        if params is not None:
            if 'params' not in list(f.attrs.keys()):
                f.attrs['params'] = ''
            f.attrs['params'] += np.str_(params)


def _get_dset_meta(fname):
    """return dataset shape and dtype"""
    fname = _split_hdf5(fname)
    with h5py.File(fname[0], 'r'):
        return f[fname[1]].shape, f[fname[1]].dtype


def load_hdf5(fname, dtype=None):
    """
    Load hdf5 data

    If fname is fed as *.h5/path or *.hdf5/path, will
    load Dataset object "path" in *.h5 or *.hdf5 file

    Args:
        fname : str, required
            filepath to load. can feed as *.h5/dataset_name
            If fed as a list, each dataset must have same size and dtype
        dtype : dtype object
            Datatype of data to load, default is ndarray dtype

    Returns:
        ndarray
            dataset
    """
    fname = _split_hdf5(fname)
    if isinstance(fname, (list, tuple)):
        shape, _dtype = _get_dset_meta(fname[0])
        if dtype is None:
            dtype = _dtype
        box = np.empty((len(fname),) + shape, dtype=dtype)
        for i, fn in enumerate(fname):
            box[i] = load_hdf5(fn, dtype=dtype)
    else:
        with h5py.File(fname, 'r') as f:
            box = f[fname[1]][:]

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



