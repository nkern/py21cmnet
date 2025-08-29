"""
neural network module
"""

import numpy as np
import torch
from torch import nn
import warnings

from . import dataset, functional


class ConvNd(nn.Module):
    """A convolution, activation and normalization block"""

    def __init__(self, conv_kwargs, conv='Conv3d',
                 activation='ReLU', act_kwargs={},
                 batch_norm='BatchNorm3d', norm_kwargs={},
                 dropout=None, dropout_kwargs={}):
        """
        A single convolutional block:
            ConvNd -> activation -> batchnorm -> dropout

        Args:
            conv_kwargs : dict, required
                keyword arguments for torch.nn.Conv2d or Conv3d
            conv : str, default = 'Conv3d'
                Convolution class, e.g. 'Conv2d' or 'Conv3d'
            activation : str, default = 'ReLU'
                activation function. None for no activation
            act_kwargs : dict, default = {}
                keyword arguments for activation function
            batch_norm : str, default = 'BatchNorm3d'
                batch normalization. None for no normalization
            norm_kwargs : dict, default = {}
                keyword arguments for batch normalization function
            dropout : str, default='Dropout3d'
                Dropout layer. None for no dropout
            dropout_kwargs : dict, default = None
                keyword args for a dropout layer. None for no dropout
        """
        super(ConvNd, self).__init__()
        steps = []
        steps.append(getattr(nn, conv)(**conv_kwargs))
        if activation is not None:
            steps.append(getattr(nn, activation)(**act_kwargs))
        if batch_norm is not None:
            steps.append(getattr(nn, batch_norm)(conv_kwargs['out_channels'], **norm_kwargs))
        if dropout is not None:
            steps.append(getattr(nn, dropout)(**dropout_kwargs))
        self.model = nn.Sequential(*steps)

    def forward(self, X):
        return self.model(X)


class UpSample(nn.Module):
    """UpSample module for decoder"""

    def __init__(self, upsample_kwargs, conv_kwargs, conv='Conv3d'):
        """
        UpSample block for decoder

        Args:
            upsample_kwargs : dict, required
                keyword arguments for nn.Upsample
            conv_kwargs : dict, required
                keyword arguments for nn.Conv2d or Conv3d
            conv : str, default = 'Conv3d'
                Convolution class, e.g. 'Conv2d' or 'Conv3d'
        """
        super(UpSample, self).__init__()
        self.model = nn.Sequential(nn.Upsample(**upsample_kwargs),
                                   getattr(nn, conv)(**conv_kwargs))

    def forward(self, X):
        return self.model(X)


class Encoder(nn.Module):
    """An encoder "conv and downsample" block"""

    def __init__(self, conv_layers, pool='MaxPool3d', pool_kwargs={},
                 device=None):
        """
        A single encoder block:
            (conv -> activation -> batchnorm -> dropout) x N -> maxpool

        Args:
            conv_layers : list of dict, required
                List of nn.Conv kwargs for each convolutional
                layer in this block
            pool : str, default = 'MaxPool3d'
                Pooling class. None for no pooling
            pool_kwargs : dict, default = {}
                kwargs for Pooling instantiation
            device : str, default=None
                device of this layer
        """
        super(Encoder, self).__init__()
        steps = []

        # append convolutional steps
        for layer in conv_layers:
            steps.append(ConvNd(**layer))
        self.model = nn.Sequential(*steps)

        # attach pooling
        if pool is not None:
            pool = getattr(nn, pool)(**pool_kwargs)
        self.pool = pool

        # send model to device if requested
        self.device = device
        if self.device is not None:
            self.to(device)

    def pass_to_device(self, X):
        """
        Pass X to device if necessary

        Args:
            X : torch.tensor
        """
        # send X to device if necessary
        if self.device is not None:
            if self.device != X.device:
                return X.to(self.device)
        return X

    def forward(self, X, pooling=True):
        """forward pass through model
        If pooling, pass through self.pool at finish
        """
        # pass through conv blocks
        out = self.model(self.pass_to_device(X))
        if pooling and self.pool is not None:
            out = self.pool(out)
        return out


class Decoder(nn.Module):
    """A decoder "upsample and conv" block"""

    def __init__(self, conv_layers, conv='Conv3d', up_kwargs={}, up_mode='upsample',
                 device=None):
        """
        A single decoder block:
            (conv -> activation -> batchnorm) x N -> upsample

        Args:
            conv_layers : list of dict, required
                List of nn.Conv kwargs for each convolutional
                layer in this block
            conv : str, default = 'Conv3d'
                Convolution class, e.g. 'Conv2d' or 'Conv3d'
            up_kwargs : dict, default = {}
                Upsampling keyword arguments
            up_mode : str, default = 'upsample'
                Upsampling method
                    'upsample'        : use UpSample (nn.Upsample, nn.ConvNd)
                    'ConvTranspose2d' : use nn.ConvTranspose2d
                    'ConvTranspose3d' : use nn.ConvTranspose3d
            device : str, default=None
                device of this layer
        """
        super(Decoder, self).__init__()
        steps = []

        # append convolutional steps
        for layer in conv_layers:
            steps.append(ConvNd(**layer))

        # append upsampling
        if up_mode == 'upsample':
            steps.append(UpSample(**up_kwargs))
        elif up_mode is None:
            pass
        elif hasattr(nn, up_mode):
            steps.append(getattr(nn, up_mode)(**up_kwargs))

        self.model = nn.Sequential(*steps)

        # send model to device if requested
        self.device = device
        if self.device is not None:
            self.to(device)

    def pass_to_device(self, X):
        """
        Pass X to device if necessary

        Args:
            X : torch.tensor
        """
        # send X to device if necessary
        if self.device is not None:
            if self.device != X.device:
                return X.to(self.device)
        return X

    def center_crop(self, X, shape):
        """
        Center crop X if needed along last Nd dimensions
        [H] and [W] denote possible but not required dimensions

        Args:
            X : torch.Tensor of shape (N, C, [H], [W], L)
            shape : len-1, 2 or 3 tuple of required cropped shape ([H], [W], L)

        Returns:
            Tensor, X center cropped to shape
        """
        Nd = len(shape)
        if X.shape[-Nd:] == shape:
            return X
        slices = []
        for i in range(1, Nd + 1):
            diff = (X.shape[-i] - shape[-i]) // 2
            if diff < 0:
                raise ValueError("Cannot center crop X of shape {} to shape {}".format(X.shape[-Nd:], shape))
            slices.append(slice(diff, diff + shape[-i]))
        if Nd == 1:
            return X[..., slices[0]]
        elif Nd == 2:
            return X[..., slices[1], slices[0]]
        elif Nd == 3:
            return X[..., slices[2], slices[1], slices[0]]

    def crop_concat(self, X, connection):
        """
        Crop and concatenate skip connection to X
        [H] and [W] denote possible but not required dimensions

        Args:
            X : torch.Tensor of shape (N, C, [H], [W], L)
            connection : torch.Tensor of shape (Nc, Cc, [Hc], [Wc], Lc)
                skip connection to center crop and concatenate to X
        """
        Nd = X.ndim - 2
        X = torch.cat([self.center_crop(connection, X.shape[-Nd:]), X], dim=1)
        return X

    def forward(self, X, connection=None, metadata=None):
        if connection is not None:
            X = self.crop_concat(X, connection)
        if metadata is not None:
            shape = X.shape
            shape[1] = len(metadata)
            X = torch.cat([X, metadata.expand(shape)], dim=1)
        out = self.model(self.pass_to_device(X))
        return out


class AutoEncoder(nn.Module):
    """An autoencoder"""

    def __init__(self, encoder_layers, decoder_layers,
                 connections=None, final_transforms=None):
        """
        An autoencoder with skip connections:
            encoder -> decoder -> final layer
              |-> skip ->|

        Args:
            encoder_layers : list of dict
                A list of Encoder kwargs for
                each encoder block in this network
            decoder_layers : list of dict
                A list of Decoder kwargs for
                each decoder block in this network
            connections : dict
                A dictionary mapping the skip connection of each
                layer index in decoder_layers to a layer index in encoder_layers.
                E.g. {1: 1, 2: 0}
            final_transforms : callable or list of callable
                Final activation to apply to each channel of network output.
                To apply a different activation for each channel, pass as a list.
        """
        super(AutoEncoder, self).__init__()
        self.connections = connections

        # setup encoder
        steps = []
        for i, encoder_kwargs in enumerate(encoder_layers):
            steps.append(Encoder(**encoder_kwargs))
        self.encoder = nn.Sequential(*steps)

        # setup decoder
        steps = []
        for i, decoder_kwargs in enumerate(decoder_layers):
            if (i + 1) == len(decoder_layers):
                if hasattr(decoder_kwargs, 'up_mode'):
                    assert decoder_kwargs['up_mode'] is None, "no upsampling after final layer"
            steps.append(Decoder(**decoder_kwargs))
        self.decoder = nn.Sequential(*steps)

        # setup activations on final layer output, one for each output channel
        if final_transforms is not None:
            # sort as Nout_chan activations
            Nout_chan = self.decoder[-1].model[-1].model[0].out_channels
            if not isinstance(final_transforms, (list, tuple)):
                final_transforms = [final_transforms for i in range(Nout_chan)]
            assert len(final_transforms) == Nout_chan

        self.final_transforms = final_transforms

    def forward(self, X, debug=False, metadata=None):
        # pass through encoder
        connects = []
        for i, encode in enumerate(self.encoder):
            # first pass thu just conv blocks to get connection
            X = encode(X, pooling=False)
            connects.append(X)
            # now pass through pooling
            if encode.pool is not None:
                X = encode.pool(connects[-1])
            if debug: print("finished encoder block {}".format(i))

        # pass through decoder
        for i, decode in enumerate(self.decoder):
            # handle connections
            if self.connections is not None and i in self.connections and self.connections[i] is not None:
                connection = connects[self.connections[i]]
            else:
                connection = None
            X = decode(X, connection, metadata=metadata if i == 0 else None)
            if debug: print("finished decoder block {}".format(i))

        # final transformations
        if self.final_transforms is not None:
            for i, ft in enumerate(self.final_transforms):
                if ft is not None:
                    X[:, i] = ft(X[:, i].clone())

        return X
