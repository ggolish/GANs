#!/usr/bin/env python3

import sys
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader

def load(optimize=True, imsize=28, batch_size=128, verbose=True):
    if imsize != 28:
        sys.stderr.write("Warning: MNIST dataset only compatible with imsize 32!\n")
    ds = torchvision.datasets.MNIST("/tmp", train=True, download=True)

    # print(ds)
    # import pprint
    # print(vars(ds))


    imgs = ((ds.data.numpy() - 127.5) / 127.5).reshape(60000, 1, 28, 28)
    return DataLoader(imgs.astype("float32"), batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    dl = load()
    batch = next(iter(dl))
    print(batch.shape, torch.min(batch), torch.max(batch))
