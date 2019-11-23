#!/usr/bin/env python3

"""
    Module for loading an image data set as a numpy array
"""

import numpy as np
import os
import sys
import argparse
import imageio
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == "loader.loader":
    from .downloader import download
else:
    from downloader import download

class GenericDataset():

    def __init__(self, ds_info: dict):
        self.ds_info = ds_info
        self.len = len([f for f in os.listdir(self.ds_info["final_dest"]) if f.endswith(".npy")])

    def __len__(self):
        return self.len

    def __getitem__(self, i: int):
        path = os.path.join(self.ds_info["final_dest"], "{}{:05d}.npy".format(self.ds_info["name"], i))
        return np.load(path)

def load_data(ds_info: dict, optimize:bool=True, verbose:bool=False, imsize:int=256, batch_size:int=128):
    
    # If the dataset is local, don't bother downloading it
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
            img = imageio.imread(path)
            images.append(img)
            if optimize:
                images.append(np.flip(img, 1))
            for img in images:
                store_img(img, count, ds_info)
                count += 1
    # else:
    #     files = [f for f in os.listdir(ds_info['final_dest']) if f.endswith('.npy')]
    #     for f in tqdm(files, ascii=True):
    #         print(f)
    if verbose:
        print('Done.')

    print(len(GenericDataset(ds_info)))

    return DataLoader(GenericDataset(ds_info), batch_size=batch_size, shuffle=True)

def store_img(img: np.array, index: int, ds_info: dict):
    # Some images may not be in the correct shape (probably black and white)
    try:
        imgnpy = np.array(img, dtype='float32')

        # Get the data ready for a pytorch GAN
        imgnpy = imgnpy / 127.5 - 1.0
        imsize, _, channels = imgnpy.shape
        imgnpy = imgnpy.reshape(channels, imsize, imsize)

        # Save each image individually
        dest = os.path.join(ds_info["final_dest"], "{}{:05d}.npy".format(ds_info["name"], index))
        np.save(dest, imgnpy)

    except ValueError:
        print('Image with bad shape')
