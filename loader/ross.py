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

data_url = "https://cs.indstate.edu/~ggolish/data/ross-data-resized.tar.gz"
data_dest = "/tmp/ross-data-resized.tar.gz"
final_dir = "/tmp/ross-data-resized"

def download_ross_data(verbose=True):
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

if __name__ == "__main__":
    download_ross_data(verbose=True)
