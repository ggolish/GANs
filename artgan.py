#!/usr/bin/env python3
import torch
from torch import nn
from architecture import dc
from loader import ross


DEFAULT_SETTINGS = {
    'arch': dc,
    'batch_size': 128,
    'image_size': 64,
    'channels': 3,
    'epochs': 1,
    'ncritic': 5,
    'grad': 10,
    'learning_rate': 0.0001,
    'beta1': 0,
    'beta2': 0.9,
    'zdim': 100
}

class Critic(nn.Module):
    """ Critic network class """

    def __init__(self, arch, channels, image_size):
        super().__init__()
        self.arch = arch.build(True, channels, image_size)
        self.channels = channels
        self.image_size = image_size

    def forward(self, x):
        return self.arch(x)

   
class Generator(nn.Module):
    """ Generator network class """

    def __init__(self, arch, channels, image_size, zdim):
        super().__init__()
        self.arch = arch.build(False, channels, image_size, nz=zdim)
        self.channels = channels
        self.image_size = image_size
        self.zdim = zdim

    def forward(self, x):
        return self.arch.forward(x)


class GAN():
    """
        Generalized GAN class
    """
    def __init__(self, dataset, settings={}):
        self.dataloader = dataset.load()
        self.G_losses = list()
        self.D_losses = list()
        iterations = 0
        self.S = DEFAULT_SETTINGS
        for k, v in settings:
            if k in self.S:
                self.S[k] = v
            else:
                sys.stderr.write(f"Warning: Invalid setting {k} = {v}!\n")
        self.D = Critic(
            self.S['arch'],
            self.S['channels'],
            self.S['image_size']
        )
        self.G = Generator(
            self.S['arch'],
            self.S['channels'],
            self.S['image_size'],
            self.S['zdim']
        )

    def train(self):
        """ Implementing a default training method from DCGAN """
        pass

    def generate_image(self, n=1):
        z = torch.normal(0, 1, (n, self.S['zdim']))
        return self.G(z)

if __name__ == '__main__':
    gan = GAN(ross)
    with torch.no_grad():
        img = gan.generate_image()

    print(gan)
    print(gan.D)
    print(gan.G)
