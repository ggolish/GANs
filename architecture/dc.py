#!/usr/bin/env python3

''' The standard DCGAN architectures for Critic and Generator. '''

import torch
import math

from torch.nn import Module, Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh, Sigmoid


class CriticArchitecture(Module):
    ''' Standard DCGAN critic architecture (base 2 image sizes only) '''

    def __init__(self, settings: dict, debug: bool = False):
        super().__init__()

        # Store necessary settings from parent GAN
        self.gp_enabled = settings['gp_enabled']
        self.imsize = settings['image_size']
        self.nchannels = settings['nchannels']
        self.nfeatures = settings['nfeatures']
        self.debug = debug

        # This architecure will only be for image sizes that are powers of two
        # (for simplicity)
        if not isbase2(self.imsize):
            raise Exception(f"Invalid image size for DCGAN: {self.imsize}")

        # Construct the necessary number of layers
        self.layers = []
        self.layers.append(
            Conv2d(self.nchannels, self.nfeatures, 4, 2, 1, bias=False))
        self.layers.append(LeakyReLU(0.02, inplace=True))
        mult = 1
        for i in range(int(math.log2(self.imsize)) - 3):
            conv2d = Conv2d(self.nfeatures * mult,
                            self.nfeatures * mult * 2, 4, 2, 1, bias=False)
            self.layers.append(conv2d)
            if not self.gp_enabled:
                bn = BatchNorm2d(self.nfeatures * mult * 2)
                self.layers.append(bn)
            activation = LeakyReLU(0.02, inplace=True)
            self.layers.append(activation)
            mult *= 2
        conv2d = Conv2d(self.nfeatures * mult, self.nfeatures *
                        mult * 2, 4, 1, 0, bias=False)
        self.layers.append(conv2d)
        self.activation = Sigmoid()

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

        if not self.gp_enabled:
            if self.debug:
                print(self.activation)
                print(x.shape, '=>', end=' ')
            x = self.activation(x)
            if self.debug:
                print(x.shape)
                print()

        return x


class GeneratorArchitecture(Module):
    ''' Standard DCGAN generator architecture (base 2 images sizes only) '''

    def __init__(self, settings: dict):
        super().__init__()


def isbase2(x):
    ''' Checks if x is a power of 2 '''
    y = math.log2(x)
    return (y - (y // 1)) == 0


if __name__ == "__main__":

    d = CriticArchitecture({
        'gp_enabled': True,
        'image_size': 64,
        'nchannels': 3,
        'nfeatures': 128
    }, debug=True)

    with torch.no_grad():
        x = torch.randn(128, 3, 64, 64)
        d(x)
