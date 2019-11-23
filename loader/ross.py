#!/usr/bin/env python3
"""
    Module for loading the Bob Ross data set as a numpy array
"""

import numpy as np

if __name__ == 'loader.ross':
    from . import downloader
    from .loader import load_data
else:
    import downloader
    from loader import load_data


ds = 'ross-data-resized'
ds_info = {
    'name': f'{ds}',
    'local_dir': f'/u1/h3/adavenport9/public_html/data/{ds}',
    'data_url': f'http://cs.indstate.edu/~ggolish/data/{ds}.tar.gz',
    'data_dest': f'/tmp/{ds}.tar.gz',
    'final_dir': f'/tmp/{ds}',
    'final_dest': f'/tmp/{ds}.npy'
}


def load(optimize=True, imsize=256, batch_size=128, verbose=True):
    global ds_info
    return load_data(ds_info, optimize=optimize, verbose=True, imsize=imsize, batch_size=batch_size)


def load_np(optimize=True, imsize=256, batch_size=128, verbose=True, nbatches=1):
    global ds_info
    dl = load_data(ds_info, optimize=optimize, verbose=True, imsize=imsize)
    li = []
    for _ in range(nbatches):
        li.extend(next(iter(dl)).numpy().tolist())
    return np.array(li)


if __name__ == '__main__':
    print('main')
    load()
