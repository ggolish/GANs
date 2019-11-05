import numpy as np


def crop(img, size):
    y, x, z = img.shape
    cx = (x // 2) - (size // 2)
    cy = (y // 2) - (size // 2)
    return img[cy:cy+size, cx:cx+size, :]
