#!/usr/bin/env python3
""" Base module for viewing images from a numpy array """

import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from loader import cubism
from loader import ross


def make_viewable(imgs: np.array):
    imgs = ((imgs * 127.5) + 127.5).astype('int16')
    n, c, y, x = imgs.shape
    return imgs.reshape(n, y, x, c)


def show_images(imgs: np.array, cols, rows, randomize=True):
    """ Take images stored in a np array and plot them. """
    imgs = make_viewable(imgs)
    if randomize:
        np.random.shuffle(imgs)
    print(imgs.shape)
    im_size = imgs.shape[2]
    grid = imgs[:cols*rows].reshape(rows, cols, im_size, im_size, 3).swapaxes(1, 2).reshape(im_size * rows, im_size * cols, 3)
    plt.axis('off')
    plt.imshow(grid, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    imgs = ross.load_np(imsize=256)
    show_images(imgs, 5, 5, randomize=False)
