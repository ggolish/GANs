#!/usr/bin/env python3
import torch
from torch import nn
from architecture.dc import DCGAN
from loader import ross


DEFAULT_SETTINGS = {
    'critic_arch': DCGAN,       # The base architucture of the critic
    'generator_arch': DCGAN,    # The base architucture of the generator
    'batch_size': 128,          # The size of each batch during training
    'image_size': 64,           # The width and height of the images (power of 2)
    'nchannels': 3,             # The number of color channels in the images
    'nfeatures': 128,           # DCGAN: The starting number of kernals in first layer
    'iterations': 1,            # The number of iterations to train on
    'ncritic': 5,               # The number of times to train critic per iteration
    'grad': 10,                 # The gradient penalty for critic
    'learning_rate': 0.0001,    # The learning rate for adam optimizer
    'beta1': 0,                 # The first beta for adam optimizer
    'beta2': 0.9,               # The second beta for adam optimizer
    'zdim': 100                 # The number of entries in the latent vectors
}

class Critic(nn.Module):
    """ Critic network class """

    def __init__(self, S=DEFAULT_SETTINGS):
        super().__init__()
        self.arch = S["critic_arch"](True, S)
        self.S = S

    def forward(self, x):
        return self.arch.forward(x)

   
class Generator(nn.Module):
    """ Generator network class """

    def __init__(self, S=DEFAULT_SETTINGS):
        super().__init__()
        self.arch = S["generator_arch"](False, S)
        self.S = S

    def forward(self, x):
        return self.arch.forward(x)


class GAN():
    """
        Generalized GAN class
    """
    def __init__(self, dataset, settings={}):
        iterations = 0
        self.S = DEFAULT_SETTINGS
        for k, v in settings.items():
            if k in self.S:
                self.S[k] = v
            else:
                sys.stderr.write(f"Warning: Invalid setting {k} = {v}!\n")
        self.D = Critic(self.S)
        self.G = Generator(self.S)
        self.dl = dataset.load(imsize=self.S["image_size"])

    def train(self, epochs):
        """ Train the network for given epochs """
        for epoch in tqdm(range(epochs)):
            pass


    def generate_image(self, n=1):
        z = torch.normal(0, 1, (n, self.S['zdim']))
        return self.G(z)

if __name__ == '__main__':
    gan = GAN(ross, {'image_size': 256})
    with torch.no_grad():
        z = torch.normal(0, 1, (1, 100))
        gan.G(z)


