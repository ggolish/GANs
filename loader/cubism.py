#!/usr/bin/env python3

"""
    Module for loading the Cubism data set
"""

import downloader

ds = 'cubism'
data_url = f'http://cs.indstate.edu/~adavenport9/data/wikiart/{ds}.tar.gz'
data_dest = f'/tmp/{ds}.tar.gz'
final_dir = f'/tmp/{ds}'


def download():
    downloader.download(ds, data_url, data_dest)


if __name__ == '__main__':
    download()
