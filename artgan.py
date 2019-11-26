#!/usr/bin/env python3

import sys
import torch
import numpy as np

from architecture import load_generator, load_critic
from torch.optim import RMSprop, Adam
from tqdm import tqdm

# The default hyperparams for a GAN
DEFAULT_SETTINGS = {
    # General hyperparams
    'critic_arch': 'dc',
    'generator_arch': 'dc',
    'zdim': 100,

    # Image hyperparams
    'image_size': 64,
    'nchannels': 3,

    # DCGAN hyperparams
    'nfeatures': 128,

    # WGAN training hyperparams
    'clipping_constant': 0.01,
    'ncritic': 5,

    # WGAN-GP training hyperparams
    'gp_enabled': False,
    'gradient_penalty': 10,
}


class GAN():

    def __init__(self, settings={}):
        ''' Generic GAN class '''

        # Parse user supplied settings
        self.S = DEFAULT_SETTINGS.copy()
        for key in settings:
            if key in self.S:
                self.S[key] = settings[key]
            else:
                sys.stderr.write(
                    f'Warning: Invalid hyperparam setting {key}!\n')

        # Choose the proper device
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build critic and generator
        self.D = load_critic(self.S).to(self.dev)
        self.G = load_generator(self.S).to(self.dev)

    def __repr__(self):
        return f'GAN(\n{repr(self.D)}\n{repr(self.G)}\n)'

    def train(self, dl, iterations=1000, lr=0.0002, si=20, bs=128):
        ''' Trains the GAN on the given data '''
        if self.S['gp_enabled']:
            yield self.train_gp(dl, iterations, lr, si, nc, bs)
        else:
            yield self.train_no_gp(dl, iterations, lr, si, nc, bs)

    def train_gp(self, dl, iterations=1000, lr=0.0002, si=20, bs=128):
        ''' Trains the GAN on the given data using wasserstein with gp'''
        pass

    def train_no_gp(self, dl, iterations=1000, lr=0.0002, si=20, bs=128):
        ''' Trains the GAN on the given data using vanilla wasserstein'''

        d_optim = RMSprop(self.D.parameters(), lr=lr)
        g_optim = RMSprop(self.D.parameters(), lr=lr)

        for iteration in tqdm(range(iterations), ascii=True):
            # Train the critic
            d_losses = []
            for _ in self.S['ncritic']:
                self.D.zero_grad()
                x_batch = next(iter(dl)).to(self.dev)
                z_batch = self.get_latent_vec(bs).to(self.dev)
                g_out = self.G(z_batch).detach()
                d_real = self.D(x_batch)
                d_fake = self.D(g_out)
                d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
                d_loss.backward()
                d_optim.step()
                d_losses.append(-d_loss.cpu().item())
                for p in self.D.parameters():
                    p.data = torch.clamp(p.data, -self.S['clipping_constant'],
                                         self.S['clipping_constant'])

            # Train the generator
            self.G.zero_grad()
            z_batch = self.get_latent_vec(bs)
            g_out = self.G(z_batch)
            d_fake = self.D(g_out)
            g_loss = -torch.mean(d_fake)
            g_loss.backward()
            g_optim.step()

            # Yield results each sample inteval
            if iteration % si == 0:
                d_avg_loss = np.mean(d_losses)
                g_loss = g_loss.cpu().item()
                yield {'d_loss': d_avg_loss, 'g_loss': g_loss}

    def get_latent_vec(self, n):
        ''' Returns random n x zdim x 1 x 1 latent vector '''
        return torch.randn(n, self.S['zdim'], 1, 1)


if __name__ == "__main__":

    gan = GAN()
    print(gan)
