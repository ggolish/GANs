#!/usr/bin/env python3
import sys
import torch
import random
from torch.nn import Module
from torch.optim import Adam, RMSProp
from architecture.dc import DCGAN
from loader import ross
from matplotlib import pyplot as plt


DEFAULT_SETTINGS = {
    'critic_arch': DCGAN,       # The base architucture of the critic
    'generator_arch': DCGAN,    # The base architucture of the generator
    'batch_size': 128,          # The size of each batch during training
    'image_size': 64,           # The width and height of the images (power of 2)
    'nchannels': 3,             # The number of color channels in the images
    'nfeatures': 128,           # DCGAN: The starting number of kernals in first layer
    'iterations': 1,            # The number of iterations to train on
    'sample_rate': 1,           # The number of iterations in which to report stats
    'ncritic': 5,               # The number of times to train critic per iteration
    'gradient_penalty': 10,     # The gradient penalty for critic
    'gp_enabled': False,        # Training with gradient penalty flag
    'learning_rate': 0.0001,    # The learning rate for adam optimizer
    'beta1': 0,                 # The first beta for adam optimizer
    'beta2': 0.9,               # The second beta for adam optimizer
    'zdim': 100,                # The number of entries in the latent vectors
    'device': "cpu"             # The device to run training on
}

class Critic(Module):
    """ Critic network class """

    def __init__(self, S=DEFAULT_SETTINGS):
        super().__init__()
        self.arch = S["critic_arch"](True, S)
        self.S = S

    def forward(self, x):
        return self.arch.forward(x)

   
class Generator(Module):
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
        self.S = DEFAULT_SETTINGS
        for k, v in settings.items():
            if k in self.S:
                self.S[k] = v
            else:
                sys.stderr.write(f"Warning: Invalid setting {k} = {v}!\n")

        if self.S["device"] == "cuda" and torch.cuda.is_available() == False:
            sys.stderr.write("Warning: Device set to cuda, cuda not available.\n")

        self.device = torch.device(self.S["device"])
        self.D = Critic(self.S).to(self.device)
        self.G = Generator(self.S).to(self.device)
        self.dl = dataset.load(batch_size=self.S["batch_size"], imsize=self.S["image_size"])

    def train(self):
        """ Train the GAN for a specified number of iterations """

        if self.S["gp_enabled"]:
            d_optim = Adam(self.D.params(), lr=self.S["learning_rate"], betas=(self.S["beta1"], self.S["beta2"]))
            g_optim = Adam(self.G.params(), lr=self.S["learning_rate"], betas=(self.S["beta1"], self.S["beta2"]))
        else:
            d_optim = RMSProp(self.D.params(), lr=self.S["learning_rate"])
            g_optim = RMSProp(self.G.params(), lr=self.S["learning_rate"])

        baseline_z = torch.normal(0, 1, (1, self.S["zdim"]))
        
        for iteration in tqdm(range(self.S["iterations"])):
            d_losses = []
            for _ in range(self.S["ncritic"]):
                x_batch = next(iter(self.dl)).to(self.device)
                z_batch = torch.normal(0, 1, (self.S["batch_size"], self.S["zdim"])).to(self.device)
                d_losses.append(self.train_critic(x_batch, z_batch, d_optim))
            z_batch = torch.normal(0, 1, (self.S["batch_size"], self.S["zdim"])).to(self.device)
            g_loss = self.train_generator(z_batch, g_optim)
            
            if iteration % self.S["sample_interval"] == 0:
                with torch.no_grad():
                    img = 127.5 * self.G(baseline_z).permute(0, 2, 3, 1).numpy() + 127.5
                yield {"d_losses": d_losses, "g_loss": g_loss, "img": img.astype("int16")}

    def train_critic(self, x_batch, z_batch, d_optim):
        self.D.zero_grad()
        if self.S["gp_enabled"]:
            pass
        else:
            pass
        return None

    def train_generator(self, z_batch, g_optim):
        self.G.zero_grad()
        return None

    def generate_image(self, n=1):
        z = torch.normal(0, 1, (n, self.S['zdim']))
        return self.G(z)

if __name__ == '__main__':
    gan = GAN(ross, {'image_size': 32, 'gp_enabled': True})


