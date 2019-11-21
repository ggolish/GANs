#!/usr/bin/env python3
""" This is the main program to run the art gan """

import torch
import argparse
from image_viewer.view_dataset import show_images
from loader import ross
from loader import cubism
from loader import impressionism


if __name__ == '__main__':
    # Probably want to manage some arguments
    parser = argparse.ArgumentParser(description='Main entry point to the GAN.')
    parser.add_argument('-d', '--dataset', action='store')
    parser.add_argument('-s', '--size', type=int, action='store')

    args = parser.parse_args()
    ds = ross
    size = 64
    """ Choosing dataset default is ross """
    if args.dataset:
        if args.dataset == 'cubism':
            ds = cubism
        elif args.dataset == 'impressionism':
            ds = impressionism

    """ Choosing image size """
    if args.size:
        # Check for that power of 2
        if args.size in [32, 64, 128, 256]:
            size = args.size

    dataset = ds.load(imsize=size)
    batch = next(iter(dataset))
    print(batch.shape, torch.min(batch), torch.max(batch))
    show_images(batch.numpy(), 5, 5)
