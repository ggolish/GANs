#!/usr/bin/env python3

"""
    Module for downloading a dataset
"""

import tarfile
from urllib.request import urlretrieve


def download(ds_info):
    print(f'Retrieving {ds_info["name"]} data set...')
    urlretrieve(ds_info['data_url'], filename=ds_info['data_dest'])
    print('Done.')
    print(f'Extracting {ds_info["name"]} data set...')
    with tarfile.open(ds_info['data_dest'], 'r') as tfd:
        tfd.extractall('/tmp')
    print('Done.')
