#!/usr/bin/env python3

'''
    Module for loading the Bob Ross data set as a numpy array
'''

import downloader

ds = 'ross-data-resized'
data_url = f'http://cs.indstate.edu/~ggolish/data/{ds}.tar.gz'
data_dest = f'/tmp/{ds}.tar.gz'
final_dir = f'/tmp/{ds}'

def download():
    downloader.download(ds, data_url, data_dest)

if __name__ == '__main__':
    download()
