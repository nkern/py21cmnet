from setuptools import setup
import os
import ast

# get version from __init__.py
init_file = os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), 'py21cmnet/__init__.py')
with open(init_file, 'r') as f:
    lines = f.readlines()
    for l in lines:
        if "__version__" in l:
            version = ast.literal_eval(l.split('=')[1].strip())

setup(
    name            = 'py21cmnet',
    version         = version,
    license         = 'MIT',
    description     = 'Deep neural networks for 21 cm fields',
    author          = 'Nicholas Kern',
    url             = "http://github.com/nkern/py21cmnet",
    include_package_data = True,
    packages        = ['py21cmnet']
    )