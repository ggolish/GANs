#!/usr/bin/env python3

"""
    Module for loading an image data set as a numpy array
"""

import imageio
import cv2
import numpy as np
import os
import sys
import tarfile
import argparse
import matplotlib.pyplot as plt
import torch
import ross

from urllib.request import urlretrieve
from tqdm import tqdm

home = os.path.expanduser("~")
data_path = home + "/research/data/images/Cubism"


def load_ross(imsize=128, batch_size=1024, verbose=True):

    if not os.path.exists(ross.final_dir):
        ross.download()
    data = load_data(ross.final_dir, verbose=True, imsize=imsize)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)


def load_data(data_dir=data_path, verbose=False, imsize=128):

    if verbose:
        print("Reading image data set...")

    files = [f for f in os.listdir(data_dir) if f.endswith(".png") or f.endswith(".jpg")]
    images = []
    for f in tqdm(files):
        path = os.path.join(data_dir, f)
        img = imageio.imread(path)
        img = cv2.imread(path)
        img = cv2.resize(img, dsize=(imsize, imsize), interpolation=cv2.INTER_CUBIC)
        images.append(img)
        images.append(np.flip(img, 1))

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
