#!/usr/bin/env python3
import numpy as np
import cv2



def crop(img: np.array, size):
    y, x, z = img.shape
    cx = (x // 2) - (size // 2)
    cy = (y // 2) - (size // 2)
    return img[cy:cy+size, cx:cx+size, :]


def northwest(img, size):
    y, x, z = img.shape
    # return np.resize(img[:y//4, :x//4, :], (size, size, z))
    img = img[:(y*2)//3, :(x*2)//3, :]
    return cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)


def southwest(img: np.array, size):
    y, x, z = img.shape
    return img[-size:, :size, :]


def southeast(img: np.array, size):
    y, x, z = img.shape
    return img[-size:, -size:, :]


def northeast(img: np.array, size):
    y, x, z = img.shape
    return img[:size, -size:, :]

if __name__ == '__main__':
    import os
    from loader import load_data
    from loader import ross
    import matplotlib.pyplot as plt

    if not os.path.exists(ross.final_dir):
        ross.download()
    data = load_data(ross.final_dir, verbose=True, imsize=256)
    data /= 255
    fig = plt.figure(figsize=(8,8))
    columns = 4
    rows = 3
    for i in range(1, columns*rows +1):
        img = data[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()
