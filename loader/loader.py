#!/usr/bin/env python3

"""
    Module for loading an image data set as a numpy array
"""

import cv2
import numpy as np
import os
import sys
import argparse
import torch
import imageio
from tqdm import tqdm

if __name__ == "loader.loader":
    from . import img_utils
    from .downloader import download
else:
    import img_utils
    from downloader import download


def create_loader(data, batch_size=128):
    return torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True)


def load_data(ds_info, optimize=True, verbose=False, imsize=256):

    if not os.path.exists(ds_info['data_dest']):
        download(ds_info)

    if verbose:
        print('Reading image data set...')

    if optimize:
        ds_info['final_dest'] = f'{ds_info["final_dir"]}.{imsize}.optimized.npy'
    else:
        ds_info['final_dest'] = f'{ds_info["final_dir"]}.{imsize}.npy'

    ds_info['final_dest'] = ds_info['final_dest'].format(imsize)

    if not os.path.exists(ds_info['final_dest']):
        files = [f for f in os.listdir(ds_info['final_dir']) if f.endswith('.png') or f.endswith('.jpg')]
        images = list()
        for f in tqdm(files, ascii=True):
            path = os.path.join(ds_info['final_dir'], f)
            original = imageio.imread(path)
            img = cv2.resize(original, dsize=(imsize, imsize), interpolation=cv2.INTER_CUBIC)
            images.append(img)
            if optimize:
                images.append(np.flip(img, 1))
                images.append(img_utils.northwest(original, imsize))
                images.append(img_utils.southwest(original, imsize))
                images.append(img_utils.southeast(original, imsize))
                images.append(img_utils.northeast(original, imsize))
        imgnpy = np.array(images, dtype='float32')

        # Get the data ready for a pytorch GAN
        imgnpy = imgnpy / 127.5 - 1.0
        n, _, _, channels = imgnpy.shape
        imgnpy = imgnpy.reshape(n, channels, imsize, imsize)
        np.save(ds_info['final_dest'], imgnpy)

    else:
        imgnpy = np.load(ds_info['final_dest'])

    if verbose:
        print('Done.')

    return imgnpy


if __name__ == "__main__":
    import ross
    parser = argparse.ArgumentParser(prog='image-loader', description='Utility for loading images')
    parser.add_argument('-s', '--size', help='Set the size of the images.')
    args = parser.parse_args(sys.argv[1:])
    if args.size:
        try:
            imsize = int(args.size)
        except ValueError:
            sys.stderr.write('Image size must be an integer.\n')
    print('Testing ross data set')
    print(ross.load())
