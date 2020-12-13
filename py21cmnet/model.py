"""
neural network models
"""
import numpy as np
import torch
from torch import nn


class Conv3d(nn.Module):
    """A convolution, activation and normalization block"""

    def __init__(self, conv_kwargs, activation='ReLU', act_kwargs={},
                 batch_norm='BatchNorm3d', norm_kwargs={}):
        """
        A single convolutional block:
            Conv3d -> activation -> batch normalization

        Args:
            conv_kwargs : dict
                keyword arguments for nn.Conv3d
            activation : str, default = 'ReLU'
                activation function. None for no activation
            act_kwargs : dict
                keyword arguments for activation function
            batch_norm : str, default = 'BatchNorm3d'
                batch normalization. None for no normalization
            norm_kwargs : dict
                keyword arguments for batch normalization function
        """
        super(Conv3d, self).__init__()
        steps = []
        steps.append(nn.Conv3d(**conv_kwargs))
        if activation is not None:
            steps.append(getattr(nn, activation)(**act_kwargs))
        if batch_norm is not None:
            steps.append(getattr(nn, batch_norm)(conv_kwargs['out_channels'], **norm_kwargs))
        self.model = nn.Sequential(*steps)

    def forward(self, X):
        return self.model(X)


class UpSample(nn.Module):
    """UpSample module for decoder"""

    def __init__(self, upsample_kwargs, conv_kwargs):
        """
        UpSample block for decoder

        Args:
            upsample_kwargs : dict
                keyword arguments for nn.Upsample
            conv_kwargs : dict
                keyword arguments for nn.Conv3d
        """
        self.model = nn.Sequential(nn.Upsample(**upsample_kwargs),
                                   nn.Conv3d(**conv_kwargs))

    def forward(self, X):
        return self.model(X)


class Encoder3d(nn.Module):
    """An encoder "conv and downsample" block"""

    def __init__(self, conv_layers, maxpool_kwargs=None):
        """
        A single encoder block:
            (conv3d -> activation -> batch norm) x N -> maxpool

        Args:
            conv_layers : list of dict
                List of Conv3d kwargs for each convolutional
                layer in this block
            maxpool_kwargs : dict
                kwargs for nn.MaxPool3d. Default is no max pooling.
        """
        super(Encoder3d, self).__init__()
        steps = []

        # append convolutional steps
        for layer in conv_layers:
            steps.append(Conv3d(**layer))

        # append max pooling
        if maxpool_kwargs is not None:
            steps.append(nn.MaxPool3d(**maxpool_kwargs))

        self.model = nn.Sequential(*steps)

    def forward(self, X):
        return self.model(X)


class Decoder3d(nn.Module):
    """A decoder "conv and upsample" block"""

    def __init__(self, conv_layers, up_kwargs, up_mode='upsample'):
        """
        A single decoder block:
            (conv -> activation -> batch norm) x N -> upsample

        Args:
            conv_layers : list of dict
                List of Conv3d kwargs for each convolutional
                layer in this block
            up_kwargs : dict
                Upsampling keyword arguments
            up_mode : str, default = 'upsample'
                Upsampling method
                    'upsample' : use UpSample (nn.Upsample, nn.Conv3d)
                    'upconv'   : use nn.ConvTranspose3d
        """
        super(Decoder3d, self).__init__()
        self.connection = connection
        steps = []

        # append convolutional steps
        for layer in conv_layers:
            steps.append(Conv3d(**layer))

        # append upsampling
        if up_mode == 'upsample':
            steps.append(UpSample(**up_kwargs))
        elif up_mode == 'upconv':
            steps.append(nn.ConvTranspose3d(**up_kwargs))
        else:
            raise ValueError("did not recognize up_mode {}".format(up_mode))

        self.model = nn.Sequential(*steps)

    def center_crop(self, X, shape):
        """
        Center crop X if needed along last three dimensions

        Args:
            X : torch.Tensor of shape (N, C, H, W, L)
            shape : len-3 tuple of required cropped shape (H, W, L)

        Returns:
            Tensor, X center cropped to shape
        """
        if X.shape[-3:] == shape:
            return X
        slices = []
        for i in range(3):
            diff = (X.shape[-i] - shape[-i]) // 2
            if diff < 0:
                raise ValueError("Cannot center crop X of shape {} to shape {}".format(X.shape[-3:], shape))
            slices.append(slice(diff, diff + shape[-i]))
        return X[..., slices[2], slices[1], slices[0]]

    def crop_concat(self, X, connection):
        """
        Crop and concatenate skip connection to X

        Args:
            X : torch.Tensor of shape (N, C, H, W, L)
            connection : torch.Tensor of shape (Nc, Cc, Hc, Wc, Lc)
                skip connection to center crop and concatenate to X
        """
        X = torch.cat([self.center_crop(connection, X.shape[-3:]), X], dim=1)
        return X

    def forward(self, X, connection=None):
        if connection is not None:
            X = self.crop_concat(X, connection)
        return self.model(X)


class AutoEncoder3d(nn.Module):
    """An autoencoder for 3D data"""

    def __init__(self, encoder_layers, decoder_layers, final_layer,
                 connections=None, final_transforms=None):
        """
        An autoencoder

        Args:
            encoder_layers : list of dict
                A list of Encoder3d kwargs for
                each encoder block in this network
            decoder_layers : list of dict
                A list of Decoder3d kwargs for
                each decoder block in this network
            final_layer : dict
                Encoder3d kwargs for the final convolutional
                layer in the network (without maxpooling)
            connections : dict
                A dictionary mapping the skip connection of each
                layer index in decoder_layers to a layer index in encoder_layers.
                E.g. {0: 2, 1: 1, 2: None}
            final_transforms : callable, list of callable
                Final activation to apply to each channel of network output
                To apply a different activation for each channel, pass as a list
        """
        super(AutoEncoder3d, self).__init__()
        self.connections = connections

        # setup encoder
        steps = []
        for encoder_kwargs in encoder_layers:
            steps.apppend(Encoder3d(**encoder_kwargs))
        self.encoder = nn.ModuleList(steps)

        # setup decoder
        steps = []
        for decoder_kwargs in decoder_layers:
            steps.append(Decoder3d(**decoder_kwargs))
        self.decoder = nn.ModuleList(steps)

        # final layer: an encoder layer with no maxpooling
        self.final = Encoder3d(final_layer, maxpool_kwargs=None)

        # setup activations on final layer output, one for each output channel
        if final_transforms is not None:
            if not isinstance(final_transforms, (list, tuple)):
                final_transforms = [final_transforms for i in range(self.final.out_channels)]
            assert len(final_transforms) == self.final.out_channels
        self.final_transforms = final_transforms

    def forward(self, X):
        # pass through encoder
        outputs = []
        for i, encode in enumerate(self.encoder):
            X = encode(X)
            outputs.append(X)  # save output for connections

        # pass through decoder
        for i, decode in enumerate(self.decoder):
            if self.connections is not None and self.connections[i] is not None:
                connection = outputs[self.connections[i]]
            else:
                connection = None
            X = decode(X, connection)

        # final layer
        out = self.final_layer(X)

        # final transformations
        if self.final_transforms is not None:
            for i, ft in enumerate(self.final_transforms):
                out[:, i] = ft(out[:, i])

        return out
