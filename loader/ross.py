#!/usr/bin/env python3
'''
    Module for loading the Bob Ross data set as a numpy array
'''

import downloader
import os
from loader.loader import load_data, create_loader

ds = 'ross-data-resized'
data_url = f'http://cs.indstate.edu/~ggolish/data/{ds}.tar.gz'
data_dest = f'/tmp/{ds}.tar.gz'
final_dir = f'/tmp/{ds}'

def download():
    downloader.download(ds, data_url, data_dest)


def load(imsize=256, batch_size=128, verbose=True):
    if not os.path.exists(final_dir):
        download()
    data = load_data(final_dir, verbose=True, imsize=imsize)
    data /= 255.0
    # For the gan things
    # data = (data / 127.5) - 1
    return create_loader(data, batch_size)


if __name__ == '__main__':
    download()
