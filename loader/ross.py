#!/usr/bin/env python3

'''
    Module for loading the Bob Ross data set as a numpy array
'''

import imageio
import numpy as np
import os
import tarfile

from urllib.request import urlretrieve
from tqdm import tqdm

data_url = "https://cs.indstate.edu/~ggolish/data/ross-data-{}.tar.gz"
data_dest = "/tmp/ross-data-{}.tar.gz"
final_dir = "/tmp/ross-data-{}"

def download_ross_data(verbose=False):
    if verbose:
        print("Retrieving Bob Ross data set...")

    urlretrieve(data_url, filename=data_dest)

    if verbose:
        print("Done.")
        print("Extracting Bob Ross data set...")

    with tarfile.open(data_dest, "r") as tfd:
        tfd.extractall("/tmp")

    if verbose:
        print("Done.")

def load_data(version="resized", verbose=False):
    global data_url, data_dest, final_dir
    data_url = data_url.format(version)
    data_dest = data_dest.format(version)
    final_dir = final_dir.format(version)

    if not os.path.exists(final_dir):
        download_ross_data(verbose=verbose)

    if verbose:
        print("Reading Bob Ross data set...")

    files = [f for f in os.listdir(final_dir) if f.endswith(".png")]
    images = []
    for f in tqdm(files):
        if f.endswith(".png"):
            path = os.path.join(final_dir, f)
            img = imageio.imread(path)
            images.append(img)

    if verbose:
        print("Done.")

    return np.array(images, dtype="float32")


if __name__ == "__main__":

    images = load_data(verbose=True)
    print("Data set shape:", images.shape)
