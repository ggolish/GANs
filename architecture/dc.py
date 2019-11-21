#!/usr/bin/env python3
"""
    Differenet architectures to be used as the generator or critic in a WGAN.
    The architectures are specific to the data being used in our experiments.
"""

import math
import torch
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, Linear, Module
from torch.nn.functional import leaky_relu, relu

if __name__ == "architecture.dc":
    from .common import Identity
else:
    from common import Identity

class DCGAN(Module):
    """ Deep Convolutional GAN architecture, for image sizes n s.t. 2^x = n where x is in N^+"""

    def __init__(self, is_critic, S):
        super().__init__()
        self.is_critic = is_critic
        self.nchannels = S['nchannels']
        self.nfeatures = S['nfeatures']
        self.zdim = S['zdim']
        self.gp_enabled = S['gp_enabled']
        
        self.final_size = S['image_size']
        for _ in range(4):
            self.final_size = convolution_size(self.final_size)
        self.fc1_size = 8 * self.nfeatures * self.final_size * self.final_size

        if is_critic:
            self.conv1 = Conv2d(self.nchannels, self.nfeatures, 3, stride=2)
            self.conv2 = Conv2d(self.nfeatures, self.nfeatures * 2, 3, stride=2)
            self.bn1 = Identity() if self.gp_enabled else BatchNorm2d(self.nfeatures * 2)
            self.conv3 = Conv2d(self.nfeatures * 2, self.nfeatures * 4, 3, stride=2)
            self.bn2 = Identity() if self.gp_enabled else BatchNorm2d(self.nfeatures * 4)
            self.conv4 = Conv2d(self.nfeatures * 4, self.nfeatures * 8, 3, stride=2)
            self.bn3 = Identity() if self.gp_enabled else BatchNorm2d(self.nfeatures * 8)
            self.fc1 = Linear(self.fc1_size, 1)
        else:
            self.fc1 = Linear(self.zdim, self.fc1_size)
            self.conv1 = ConvTranspose2d(8 * self.nfeatures, 4 * self.nfeatures, 3, stride=2)
            self.bn1 = BatchNorm2d(4 * self.nfeatures)
            self.conv2 = ConvTranspose2d(4 * self.nfeatures, 2 * self.nfeatures, 3, stride=2)
            self.bn2 = BatchNorm2d(2 * self.nfeatures)
            self.conv3 = ConvTranspose2d(2 * self.nfeatures, self.nfeatures, 3, stride=2)
            self.bn3 = BatchNorm2d(self.nfeatures)
            self.conv4 = ConvTranspose2d(self.nfeatures, self.nchannels, 4, stride=2)

    def forward(self, x):
        if self.is_critic:
            x = leaky_relu(self.conv1(x))
            x = leaky_relu(self.bn1(self.conv2(x)))
            x = leaky_relu(self.bn2(self.conv3(x)))
            x = leaky_relu(self.bn3(self.conv4(x)))
            x = self.fc1(x.view(-1, self.fc1_size))
        else:
            x = self.fc1(x).view(-1, 8 * self.nfeatures, self.final_size, self.final_size)
            x = relu(self.bn1(self.conv1(x)))
            x = relu(self.bn2(self.conv2(x)))
            x = relu(self.bn3(self.conv3(x)))
            x = torch.tanh(self.conv4(x))
        return x


def convolution_size(imsize, kernal_size=3, stride=2, padding=0, dilation=1):
    """ Helper function to calculate image size after a convolution. """
    return math.floor((imsize + 2 * padding - dilation * (kernal_size - 1) - 1) / stride + 1)

