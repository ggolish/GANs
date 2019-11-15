#!/usr/bin/env python3
'''
    Differenet architectures to be used as the generator or critic in a WGAN.
    The architectures are specific to the data being used in our experiments.
'''

from torch.nn import Conv2d, Conv2dTranspose, BatchNorm2d, 
from torch.nn.functional import leaky_relu

class DC():
    ''' Deep Convolutional GAN architecture, for image sizes n s.t. 2^x = n where x is in N^+'''

    def __init__(self, is_critic, S):
        self.nchannels = S["nchannels"]
        self.nfeatures = S["nfeatures"]
        self.zdim = S["zdim"]

        if is_critic:
            self.conv1 = Conv2d(self.nchannels, self.nfeatures, kernal_size=3, stride=2)
            self.conv2 = Conv2d(self.nfeatures * 2, self.nfeatures * 4, kernal_size=3, stride=2)
            self.conv3 = Conv2d(self.nfeatures * 4, self.nfeatures * 8, kernal_size=3, stride=2)
            self.conv4 = Conv2d(self.nfeatures * 8, 1, kernal_size=3, stride=1)
        else:
            pass
        pass

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)

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
