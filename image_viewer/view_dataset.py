#!/usr/bin/env python3
""" Base module for viewing images from a numpy array """

import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from loader import cubism


def show_images(imgs: np.array, cols, rows, shuffle=True):
    """ Take images stored in a np array and plot them. """
    shuffle(imgs)
    fig=plt.figure(figsize=(8, 8))
    for i in range(1, cols*rows +1):
        fig.add_subplot(rows, columns, i)
        # Have to do i-1 due to the indexes here
        plt.imshow(imgs[i-1])
    plt.show()

if __name__ == '__main__':
    imgs = cubism.load_np
    show_images(imgs, 10, 10)
