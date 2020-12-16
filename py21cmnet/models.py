"""
neural network module
"""

import numpy as np
import torch
from torch import nn
import warnings


class ConvNd(nn.Module):
    """A convolution, activation and normalization block"""

    def __init__(self, conv_kwargs, conv='Conv3d',
                 activation='ReLU', act_kwargs={},
                 batch_norm='BatchNorm3d', norm_kwargs={},
                 dropout='Dropout3d', dropout_kwargs={}):
        """
        A single convolutional block:
            ConvNd -> activation -> batch normalization (-> dropout)

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

    def __init__(self, conv_layers, maxpool='MaxPool3d', maxpool_kwargs={}, device=None):
        """
        A single encoder block:
            (conv -> activation -> batch norm) x N -> maxpool

        Args:
            conv_layers : list of dict, required
                List of nn.Conv kwargs for each convolutional
                layer in this block
            maxpool : str, default = 'MaxPool3d'
                Maxpooling class. None for no maxpooling
            maxpool_kwargs : dict, default = {}
                kwargs for nn.MaxPool
            device : str, default=None
                device of this layer
        """
        super(Encoder, self).__init__()
        steps = []

        # append convolutional steps
        for layer in conv_layers:
            steps.append(ConvNd(**layer))

        # append max pooling
        if maxpool is not None:
            steps.append(getattr(nn, maxpool)(**maxpool_kwargs))

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

    def forward(self, X):
        return self.model(self.pass_to_device(X))


class Decoder(nn.Module):
    """A decoder "conv and upsample" block"""

    def __init__(self, conv_layers, up_kwargs, conv='Conv3d', up_mode='upsample', device=None):
        """
        A single decoder block:
            (conv -> activation -> batch norm) x N -> upsample

        Args:
            conv_layers : list of dict, required
                List of nn.Conv kwargs for each convolutional
                layer in this block
            up_kwargs : dict, required
                Upsampling keyword arguments
            conv : str, default = 'Conv3d'
                Convolution class, e.g. 'Conv2d' or 'Conv3d'
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
        else:
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

    def forward(self, X, connection=None):
        if connection is not None:
            X = self.crop_concat(X, connection)
        return self.model(self.pass_to_device(X))


class AutoEncoder(nn.Module):
    """An autoencoder"""

    def __init__(self, encoder_layers, decoder_layers, final_layer,
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
            final_layer : dict
                Encoder kwargs for the final convolutional
                layer in the network (without maxpooling)
            connections : dict
                A dictionary mapping the skip connection of each
                layer index in decoder_layers to a layer index in encoder_layers.
                E.g. {0: 2, 1: 1, 2: None}
            final_transforms : callable, list of callable
                Final activation to apply to each channel of network output
                To apply a different activation for each channel, pass as a list
        """
        super(AutoEncoder, self).__init__()
        self.connections = connections

        # setup encoder
        steps = []
        for encoder_kwargs in encoder_layers:
            steps.append(Encoder(**encoder_kwargs))
        self.encoder = nn.ModuleList(steps)

        # setup decoder
        steps = []
        for decoder_kwargs in decoder_layers:
            steps.append(Decoder(**decoder_kwargs))
        self.decoder = nn.ModuleList(steps)

        # final layer: an encoder layer with no maxpooling
        self.final = Encoder(**final_layer, maxpool=None)

        # setup activations on final layer output, one for each output channel
        if final_transforms is not None:
            if not isinstance(final_transforms, (list, tuple)):
                final_transforms = [final_transforms for i in range(self.final.out_channels)]
            assert len(final_transforms) == self.final.model[-1].model[0].out_channels 
        self.final_transforms = final_transforms

    def forward(self, X):
        # pass through encoder
        outputs = []
        for i, encode in enumerate(self.encoder):
            X = encode(X)
            outputs.append(X)  # save output for connections

        # pass through decoder
        for i, decode in enumerate(self.decoder):
            if self.connections is not None and i in self.connections and self.connections[i] is not None:
                connection = outputs[self.connections[i]]
            else:
                connection = None
            X = decode(X, connection)

        # final layer
        out = self.final(X)

        # final transformations
        if self.final_transforms is not None:
            for i, ft in enumerate(self.final_transforms):
                out[:, i] = getattr(nn, ft)()(out[:, i].clone())

        return out


class ModifiedTanh(nn.Module):
    r"""Modified Tanh - i.e. a sharper sigmoid

    :math:`\text{MTanh}(x) = \text{Tanh}(x) / 2 + 0.5`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    def __init__(self, inplace=False):
        super(ModifiedTanh, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return nn.functional.tanh(input, inplace=self.inplace) / 2 + 0.5


class ModifiedHardtanh(nn.Module):
    r"""Modified Hardtanh - i.e. a sharper Hard sigmoid

    :math:`\text{MHardtanh}(x) = \text{Hardtanh}(x) / 2 + 0.5`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    def __init__(self, inplace=False):
        super(ModifiedHardtanh, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return nn.functional.hardtanh(input, inplace=self.inplace) / 2 + 0.5
