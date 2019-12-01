#!/usr/bin/env python3
""" Module to create and load the static z vectors for comparision """

import torch
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
Z_PATH = os.path.join(dir_path,'Z.pt')


def load_static(n=16):
    """ Get a static z with n elements """
    if n > 1024:
        sys.stderr.write('Requested too large of a Z')
        n = 1024
    return torch.load(Z_PATH)[:n]


if __name__ == '__main__':
    z = load_static()
    print(z.shape)
    print(z[0, 0])
