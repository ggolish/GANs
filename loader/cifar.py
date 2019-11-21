#!/usr/bin/env python3

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

def load(batch_size=128, class_type="cat"):
    global classes
    ds = torchvision.datasets.CIFAR10("/tmp", train=False, download=True)
    indeces = [i for i in range(len(ds)) if ds[i][1] == classes[class_type]]
    imgs = np.transpose((ds.data[indeces] - 127.5) / 127.5, (0, 3, 1, 2))
    return DataLoader(imgs, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    dl = load()
    batch = next(iter(dl))
    print(batch.shape, torch.min(batch), torch.max(batch))
