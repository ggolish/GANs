#!/usr/bin/env python3

import sys
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader

classes = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}

def load(class_type="horse", optimize=True, imsize=32, batch_size=128, verbose=True):
    global classes
    if imsize != 32:
        sys.stderr.write("Warning: CIFAR10 dataset only compatible with imsize 32!\n")
    ds = torchvision.datasets.CIFAR10("/tmp", train=True, download=True)
    indeces = [i for i in range(len(ds)) if ds[i][1] == classes[class_type]]
    imgs = (ds.data[indeces].transpose(0, 3, 1, 2) - 127.5) / 127.5
    return DataLoader(imgs.astype("float32"), batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    dl = load()
    batch = next(iter(dl))
    print(batch.shape, torch.min(batch), torch.max(batch))
