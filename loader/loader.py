#!/usr/bin/env python3

"""
    Module for loading an image data set as a numpy array
"""

import numpy as np
import os
import sys
import argparse
import imageio
import skimage
from torch.utils.data import DataLoader
from skimage.transform import resize
from tqdm import tqdm

if __name__ == "loader.loader":
    from . import img_utils
    from .downloader import download
else:
    import img_utils
    from downloader import download

class GenericDataset():

    def __init__(self, ds_info):
        self.ds_info = ds_info
        self.len = len([f for f in os.listdir(self.ds_info["final_dest"]) if f.endswith(".npy")])

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        path = os.path.join(self.ds_info["final_dest"], "{}{:05d}.npy".format(self.ds_info["name"], i))
        return np.load(path)

def load_data(ds_info, optimize=True, verbose=False, imsize=256, batch_size=128):

    if os.path.exists(ds_info['local_dir']):
        ds_info['final_dir'] = ds_info['local_dir']
        print(ds_info['final_dir'])

    elif not os.path.exists(ds_info['data_dest']):
        download(ds_info)

    if verbose:
        print('Reading image data set...')

    if optimize:
        ds_info['final_dest'] = f'{ds_info["final_dir"]}.{imsize}.optimized'
    else:
        ds_info['final_dest'] = f'{ds_info["final_dir"]}.{imsize}'

    if not os.path.exists(ds_info['final_dest']):
        os.mkdir(ds_info["final_dest"])
        files = [f for f in os.listdir(ds_info['final_dir']) if f.endswith('.png') or f.endswith('.jpg')]
        count = 0
        for f in tqdm(files, ascii=True):
            images = []
            path = os.path.join(ds_info['final_dir'], f)
            original = imageio.imread(path)
            img = resize(original, (imsize, imsize, 3), cval=3, preserve_range=True, anti_aliasing=True)
            images.append(img)
            if optimize:
                images.append(np.flip(img, 1))
                images.append(img_utils.northwest(original, imsize))
                images.append(img_utils.southwest(original, imsize))
                images.append(img_utils.southeast(original, imsize))
                images.append(img_utils.northeast(original, imsize))
            for img in images:
                store_img(img, count, ds_info)
                count += 1

    if verbose:
        print('Done.')

    return DataLoader(GenericDataset(ds_info), batch_size=batch_size, shuffle=True)

def store_img(img, index, ds_info):
    imgnpy = np.array(img, dtype='float32')

    # Get the data ready for a pytorch GAN
    imgnpy = imgnpy / 127.5 - 1.0
    imsize, _, channels = imgnpy.shape
    imgnpy = imgnpy.reshape(channels, imsize, imsize)

    # Save each image individually
    dest = os.path.join(ds_info["final_dest"], "{}{:05d}.npy".format(ds_info["name"], index))
    np.save(dest, imgnpy)

