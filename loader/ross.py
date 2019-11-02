#!/usr/bin/env python3

'''
    Module for loading the Bob Ross data set as a numpy array
'''

import downloader

ds = 'ros-data-resized'
data_url = f'http://cs.indstate.edu/~adavenport9/data/wikiart/{ds}.tar.gz'
data_dest = f'/tmp/{ds}.tar.gz'
final_dir = f'/tmp/{ds}'

def download():
    downloader.download_data(data_url, data_dest)
