# py21cmnet
![Run Tests](https://github.com/nkern/py21cmnet/workflows/Run%20Tests/badge.svg)
[![codecov](https://codecov.io/gh/nkern/py21cmnet/branch/main/graph/badge.svg?token=3Q1IZUGZ5W)](https://codecov.io/gh/nkern/py21cmnet)

A deep convolutional autoencoder network for 21 cm fields, built on pytorch.

## Installation

Clone this repository as

`git clone https://github.com/nkern/py21cmnet`

`cd` into the directory and install as

`pip install -e .`

or

`python setup.py install`

### Dependencies

Major `pip` or `conda` installable dependencies include:

* torch>=1.7.0
* torchvision>=0.8.1
* numpy>=1.18
* scipy>=1.4.0
* scikit-learn
* scikit-image
* pyyaml
* h5py

## Getting Started

To build an auto-encoder, specify the network parameters using a YAML configuration file
following the examples in `py21cmnet/config`.

```python
import os
import torch
from py21cmnet import models, utils, dataset
from py21cmnet.data import DATA_PATH
from py21cmnet.config import CONFIG_PATH

# load a model
params = utils.load_autoencoder_params(os.path.join(CONFIG_PATH, "autoencoder2d.yaml"),
                                       os.path.join(CONFIG_PATH, "autoencoder2d_defaults.yaml"))
model = models.AutoEncoder(**params)

# load a dataset
fname = os.path.join(DATA_PATH, "train_21cmfast_basic.h5")
X, y = utils.read_test_data(fname, ndim=2)

# evaluate the data
out = model(X)

# train
ds = dataset.BoxDataset(X, y, utils.load_dummy, transform=dataset.Roll(ndim=2))
dl = torch.utils.data.DataLoader(ds)
info = utils.train(model, dl, torch.nn.MSELoss(reduction='mean'), torch.optim.Adam,
                   optim_kwargs=dict(lr=0.1), Nepochs=3)
```


