#!/usr/bin/env python3
""" Module to create and load the static z vectors for comparision """

import torch
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
Z_PATH = os.path.join(dir_path,'Z.pt')
loaded_z = None


def load_static(n=16):
    """ Get a static z with n elements """
    global loaded_z
    if n > 1024:
        sys.stderr.write('Requested too large of a Z')
        n = 1024
    if loaded_z is None:
        loaded_z = torch.load(Z_PATH)[:n]
    return loaded_z


if __name__ == '__main__':
    z = load_static()
    print(z.shape)
    print(z[0, 0])
