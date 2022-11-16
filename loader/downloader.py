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
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tfd, "/tmp")
    print('Done.')
