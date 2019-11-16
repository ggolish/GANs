#!/usr/bin/env python3

"""
    Module for loading the Cubism data set
"""
if __name__ == "loader.cubism":
    from . import downloader
    from .loader import load_data, create_loader
else:
    import downloader
    from loader import load_data, create_loader

import os

ds = "cubism"
ds_info = {
    "name": ds,
    "data_url": f'http://cs.indstate.edu/~adavenport9/data/wikiart/{ds}.tar.gz',
    "data_dest": f'/tmp/{ds}.tar.gz',
    "final_dir": f'/tmp/{ds}',
    "final_dest": f'/tmp/{ds}.npy'
}


def load(optimize=True, imsize=256, batch_size=128, verbose=True):
    global ds_info
    data = load_data(ds_info, optimize=optimize, verbose=True, imsize=imsize)
    return create_loader(data, batch_size)


def load_np(optimize=True, imsize=256, batch_size=128, verbose=True):
    global ds_info
    return load_data(ds_info, optimize=optimize, verbose=True, imsize=imsize)


if __name__ == '__main__':
    load()
