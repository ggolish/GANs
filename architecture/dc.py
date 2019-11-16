#!/usr/bin/env python3
'''
    Differenet architectures to be used as the generator or critic in a WGAN.
    The architectures are specific to the data being used in our experiments.
'''

import math

from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, Linear
from torch.nn.functional import leaky_relu

class DCGAN():
    ''' Deep Convolutional GAN architecture, for image sizes n s.t. 2^x = n where x is in N^+'''

    def __init__(self, is_critic, S):
        self.is_critic = is_critic
        self.nchannels = S["nchannels"]
        self.nfeatures = S["nfeatures"]
        self.zdim = S["zdim"]

        if is_critic:
            curr_size = S["image_size"]
            self.conv1 = Conv2d(self.nchannels, self.nfeatures, 3, stride=2)
            curr_size = convolution_size(curr_size)
            self.conv2 = Conv2d(self.nfeatures, self.nfeatures * 2, 3, stride=2)
            curr_size = convolution_size(curr_size)
            self.conv3 = Conv2d(self.nfeatures * 2, self.nfeatures * 4, 3, stride=2)
            curr_size = convolution_size(curr_size)
            self.conv4 = Conv2d(self.nfeatures * 4, self.nfeatures * 8, 3, stride=2)
            curr_size = convolution_size(curr_size)
            self.fc1_size = 8 * self.nfeatures * curr_size * curr_size
            self.fc1 = Linear(self.fc1_size, 1)
        else:
            pass
        pass

    def forward(self, x):
        if self.is_critic:
            print(x.shape)
            x = self.conv1(x)
            print(x.shape)
            x = self.conv2(x)
            print(x.shape)
            x = self.conv3(x)
            print(x.shape)
            x = self.conv4(x)
            print(x.shape)
            x = self.fc1(x.view(-1, self.fc1_size))
            print(x.shape)
        else:
            pass
        return x


def convolution_size(imsize, kernal_size=3, stride=2, padding=0, dilation=1):
    ''' Helper function to calculate image size after a convolution. '''
    return math.floor((imsize + 2 * padding - dilation * (kernal_size - 1) - 1) / stride + 1)

def build(isCritic, nc, nf, nz=100):
    """
        isCritic: 
        nc : number of channels
        nf : number of features
        nz : optional size of latent vector
    """
    if isCritic:
         return nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf) x 32 x 32
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf*2) x 16 x 16
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf*4) x 8 x 8
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf*8) x 4 x 4
            nn.Conv2d(nf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    else:
        return nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, nf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),
            # state size. (nf*8) x 4 x 4
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            # state size. (nf*4) x 8 x 8
            nn.ConvTranspose2d( nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            # state size. (nf*2) x 16 x 16
            nn.ConvTranspose2d( nf * 2, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            # state size. (nf) x 32 x 32
            nn.ConvTranspose2d( nf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
