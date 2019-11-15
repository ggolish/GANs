#!/usr/bin/env python3
""" This is the main program to run the art gan """

from loader import ross
import torch


if __name__ == '__main__':
    # Probably want to manage some arguments
    dl = ross.load()
    batch = next(iter(dl))
    print(batch.shape, torch.min(batch), torch.max(batch))
