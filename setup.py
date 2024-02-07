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

def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths


data_files = package_files('py21cmnet', 'data') + package_files('py21cmnet', 'config')

setup(
    name            = 'py21cmnet',
    version         = version,
    license         = 'MIT',
    description     = 'Deep neural networks for 21 cm fields',
    author          = 'Nicholas Kern',
    url             = "http://github.com/nkern/py21cmnet",
    package_data    = {'py21cmnet': data_files},
    include_package_data = True,
    packages        = ['py21cmnet'],
    package_dir     = {'py21cmnet': 'py21cmnet'},
    zip_safe        = False
    )