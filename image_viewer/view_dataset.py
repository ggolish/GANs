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
    imgs = (imgs + 1) / 2
    n, c, y, x = imgs.shape
    return imgs.reshape(n, y, x, c)


def show_images(imgs: np.array, cols, rows, randomize=True):
    """ Take images stored in a np array and plot them. """
    imgs = make_viewable(imgs)
    if randomize:
        np.random.shuffle(imgs)
    fig = plt.figure(figsize=(256, 256))
    for i in range(1, cols*rows + 1):
        fig.add_subplot(rows, cols, i)
        # Have to do i-1 due to the indexes here
        plt.imshow(imgs[i-1])
    plt.axis('off')
    plt.axis('tight')
    plt.axis('image')
    plt.show()


if __name__ == '__main__':
    imgs = ross.load_np(imsize=256)
    show_images(imgs, 5, 5)
