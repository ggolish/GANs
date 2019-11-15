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

ds = 'cubism'
data_url = f'http://cs.indstate.edu/~adavenport9/data/wikiart/{ds}.tar.gz'
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
    # return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return create_loader(data, batch_size)


if __name__ == '__main__':
    download()
