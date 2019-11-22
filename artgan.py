#!/usr/bin/env python3
import sys
import torch
import random
import numpy as np
import trainer
from torch.nn import Module
from torch.optim import Adam, RMSprop
from architecture.dc import DCGAN
from loader import ross, cifar
from matplotlib import pyplot as plt
from tqdm import tqdm


DEFAULT_SETTINGS = {
    'critic_arch': DCGAN,       # The base architucture of the critic
    'generator_arch': DCGAN,    # The base architucture of the generator
    'batch_size': 128,          # The size of each batch during training
    'image_size': 64,           # The width and height of the images (power of 2)
    'nchannels': 3,             # The number of color channels in the images
    'nfeatures': 128,           # DCGAN: The starting number of kernals in first layer
    'iterations': 1,            # The number of iterations to train on
    'sample_interval': 1,       # The number of iterations in which to report stats
    'ncritic': 5,               # The number of times to train critic per iteration
    'clipping': '0.01',         # The clipping constant for wasserstein distance (gp_enabled == false)
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

    def parameters(self):
        return self.arch.parameters()

   
class Generator(Module):
    """ Generator network class """

    def __init__(self, S=DEFAULT_SETTINGS):
        super().__init__()
        self.arch = S["generator_arch"](False, S)
        self.S = S

    def forward(self, x):
        return self.arch.forward(x)

    def parameters(self):
        return self.arch.parameters()


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
        self.gen_mode = (dataset == None)
        if not self.gen_mode:
            self.dl = dataset.load(batch_size=self.S["batch_size"], imsize=self.S["image_size"])

    def train(self):
        """ Train the GAN for a specified number of iterations """

        if self.gen_mode:
            sys.stderr.write("Error: GAN loaded in generate mode, cannot train.")
            return 

        if self.S["gp_enabled"]:
            d_optim = Adam(self.D.parameters(), lr=self.S["learning_rate"], betas=(self.S["beta1"], self.S["beta2"]))
            g_optim = Adam(self.G.parameters(), lr=self.S["learning_rate"], betas=(self.S["beta1"], self.S["beta2"]))
        else:
            d_optim = RMSprop(self.D.parameters(), lr=self.S["learning_rate"])
            g_optim = RMSprop(self.G.parameters(), lr=self.S["learning_rate"])

        baseline_z = torch.normal(0, 1, (1, self.S["zdim"]))
        
        for iteration in tqdm(range(self.S["iterations"]), ascii=True):
            d_losses = []
            for _ in range(self.S["ncritic"]):
                x_batch = next(iter(self.dl)).to(self.device)
                z_batch = torch.normal(0, 1, (self.S["batch_size"], self.S["zdim"])).to(self.device)
                curr_loss = self.train_critic(x_batch, z_batch, d_optim)
                d_losses.append(curr_loss.item())
            z_batch = torch.normal(0, 1, (self.S["batch_size"], self.S["zdim"])).to(self.device)
            g_loss = self.train_generator(z_batch, g_optim).item()
            
            if iteration % self.S["sample_interval"] == 0:
                with torch.no_grad():
                    img = 127.5 * self.G(baseline_z).permute(0, 2, 3, 1).numpy() + 127.5
                yield {"d_losses": np.array(d_losses), "g_loss": g_loss, "img": img.astype("int16")}

    def train_critic(self, x_batch, z_batch, d_optim):
        if self.gen_mode:
            sys.stderr.write("Error: GAN loaded in generate mode, cannot train.")
            return 
        self.D.zero_grad()
        loss = None
        if self.S["gp_enabled"]:
            pass
        else:
            y_real = self.D(x_batch)
            y_fake = self.D(self.G(z_batch).detach())
            loss = torch.mean(y_fake) - torch.mean(y_real)
        loss.backward()
        d_optim.step()
        if self.S["gp_enabled"]:
            for p in seld.D.parameters():
                p.data = torch.clamp(p.data, -self.S["clipping"], self.S["clipping"])
        return loss

    def train_generator(self, z_batch, g_optim):
        if self.gen_mode:
            sys.stderr.write("Error: GAN loaded in generate mode, cannot train.")
            return 
        self.G.zero_grad()
        y_fake = self.D(self.G(z_batch))
        loss = -torch.mean(y_fake)
        loss.backward()
        g_optim.step()
        return loss

    def generate_image(self, n=1):
        z = torch.normal(0, 1, (n, self.S['zdim']))
        return self.G(z)

if __name__ == '__main__':
    gan = GAN(cifar, {
        'image_size': 32, 
        'nchannels': 3,
        'iterations': 500,
        'sample_interval': 50
    })

    trainer.train(gan, "cifar-test")

