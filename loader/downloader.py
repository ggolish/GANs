#!/usr/bin/env python3

"""
    Module for downloading a dataset
"""

import tarfile
from urllib.request import urlretrieve


def download(ds, data_url, data_dest):
    print(f"Retrieving {ds} data set...")
    urlretrieve(data_url, filename=data_dest)
    print("Done.")
    print("Extracting {ds} data set...")
    with tarfile.open(data_dest, "r") as tfd:
        tfd.extractall("/tmp")
    print("Done.")
