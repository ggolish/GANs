#!/usr/bin/env python3

import math

def dconv2d_sizes(imsize):
    kernal_size = 3 
    stride = 2
    padding = 1
    dilation = 1
    output_padding = 0
    nf = 1024
    print(imsize)
    while nf >= 128:
        imsize = (imsize - 1) * stride - 2 * padding + dilation * (kernal_size - 1) + output_padding + 1
        print(imsize)
        nf /= 2

def conv2d_sizes(imsize):
    kernal_size = 3
    stride = 2
    padding = 1
    dilation = 1
    nf = 128
    print(imsize)
    while nf <= 1024:
        imsize = math.floor((imsize + 2 * padding - dilation * (kernal_size - 1) - 1) / stride + 1)
        print(imsize)
        nf *= 2
    return imsize

if __name__ == "__main__":
    import sys

    print("Convolution:")
    imsize = conv2d_sizes(float(sys.argv[1]))
    print()
    print("Deconvolution:")
    dconv2d_sizes(imsize)
