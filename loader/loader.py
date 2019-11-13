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
import ross
import cubism
import img_utils
from tqdm import tqdm

home = os.path.expanduser("~")
data_path = home + "/research/data/images/Cubism"


def load_ross(imsize=256, batch_size=128, verbose=True):
    if not os.path.exists(ross.final_dir):
        ross.download()
    data = load_data(ross.final_dir, verbose=True, imsize=imsize)
    data /= 255.0
    # For the gan things
    # data = (data / 127.5) - 1
    # return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return create_loader(data, batch_size)


def load_cubism(imsize=256, batch_size=128, verbose=True):
    if not os.path.exists(cubism.final_dir):
        cubism.download()
    data = load_data(cubism.final_dir, verbose=True, imsize=imsize)
    data /= 255.0
    # For the gan things
    # data = (data / 127.5) - 1
    # return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return create_loader(data, batch_size)


def create_loader(data, batch_size=128):
    return torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True)


def load_data(data_dir=data_path, optimize=True, verbose=False, imsize=256):

    if verbose:
        print("Reading image data set...")

    files = [f for f in os.listdir(data_dir) if f.endswith(".png") or f.endswith(".jpg")]
    images = list()
    for f in tqdm(files[:6]):
        path = os.path.join(data_dir, f)
        original = cv2.imread(path)
        img = cv2.resize(original, dsize=(imsize, imsize), interpolation=cv2.INTER_CUBIC)
        print(type(original))
        images.append(img)
        if optimize:
            images.append(np.flip(img, 1))
            images.append(img_utils.northwest(original, imsize))
            images.append(img_utils.southwest(original, imsize))
            images.append(img_utils.southeast(original, imsize))
            images.append(img_utils.northeast(original, imsize))

    if verbose:
        print("Done.")

    return np.array(images, dtype="float32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="image-loader", description="Utility for loading images")
    parser.add_argument("-d", "--directory", help="Set the data directory.")
    parser.add_argument("-s", "--size", help="Set the size of the images.")
    args = parser.parse_args(sys.argv[1:])

    if args.directory:
        if not os.path.exists(args.directory):
            sys.stderr.write('Specify a valid directory.\n')
        else:
            data_path = args.directory

    if args.size:
        try:
            imsize = int(args.size)
        except ValueError:
            sys.stderr.write('Image size must be an integer.\n')
    print('Testing ross data set')
    print(load_ross())
