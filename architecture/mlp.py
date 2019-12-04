#!/usr/bin/env python3

''' Multi-layer perceptron architecture for Critic and Generator. '''

import torch
import math

from torch.nn import Module, Linear, ConvTranspose2d, BatchNorm2d
from torch.nn import ReLU, Tanh, Sigmoid, LeakyReLU


class CriticArchitecture(Module):
    ''' Standard MLP critic architecture (base 2 image sizes only) '''

    def __init__(self, settings: dict, debug: bool = False):
        super().__init__()

        # Store necessary settings from parent GAN
        self.imsize = settings['image_size']
        self.nchannels = settings['nchannels']
        # self.nfeatures = settings['nfeatures']
        # self.gp_enabled = settings['gp_enabled']
        self.layer_size = settings['layer_size']
        self.debug = debug
        self.input_size = (imsize**2) * nchannels

        # This architecure will only be for image sizes that are powers of two
        # (for simplicity)
        if not isbase2(self.imsize):
            raise Exception(f"Invalid image size for DCGAN: {self.imsize}\nMust be a power of two.")

        # Construct the necessary number of layers
        self.layers = []
        self.layers.append(
            Linear(self.input_size, self.layer_size)
        )
        # Going to try leakyrelu activation
        self.layers.append(LeakyReLU(0.02, inplace=True))

        self.layers.append(
            Linear(self.layer_size,1)
        )
        self.layers.append(LeakyReLU(0.02, inplace=True))

        # Ensure all parameters are accessible
        for i, layer in enumerate(self.layers):
            self.add_module(f'layer{i}', layer)

    def forward(self, x):
        if self.debug:
            print("DCGAN Critic Forward Method:\n")
        for layer in self.layers:
            if self.debug:
                print(layer)
                print(x.shape, '=>', end=' ')
            x = layer(x)
            if self.debug:
                print(x.shape)
                print()

        return x


class GeneratorArchitecture(Module):
    ''' MLP generator architecture (base 2 images sizes only) '''

    def __init__(self, settings: dict, debug=False):
        super().__init__()

        # Store necessary parameters from parent GAN
        self.zdim = settings['zdim']
        # self.nfeatures = settings['nfeatures']
        self.nchannels = settings['nchannels']
        self.imsize = settings['image_size']
        self.debug = debug
        self.layer_size = settings['layer_size']
        # Need to calculate output size
        self.output_size = (self.imsize**2) * self.nchannels

        # Ensure image is a power of 2
        if not isbase2(self.imsize):
            raise Exception(
                f"Invalid image size for DCGAN generator: {self.imsize}\nMust be a power of 2.")

        # Build appropriate number of layers
        self.layers = []
        self.layers.append(Linear(self.zdim, self.layer_size))

        self.layers.append(Linear(self.layer_size, self.output_size))
        self.layers.append(Tanh())

        # Ensure parameters are accessible
        for i, layer in enumerate(self.layers):
            self.add_module(f'layer{i}', layer)

    def forward(self, x):
        if self.debug:
            print("DCGAN Generator Forward Method:\n")
        for layer in self.layers:
            if self.debug:
                print(layer)
                print(x.shape, '=>', end=' ')
            x = layer(x)
            if self.debug:
                print(x.shape)
                print()
        return x


def isbase2(x):
    ''' Checks if x is a power of 2 '''
    y = math.log2(x)
    return (y - (y // 1)) == 0


if __name__ == "__main__":

    g = GeneratorArchitecture({
        'image_size': 32,
        'nchannels': 3,
        'zdim': 100,
        'layer_size': 512
    }, debug=True)

    with torch.no_grad():
        z = torch.randn(100, 1, 1)
        g(z.flatten())
