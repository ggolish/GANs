#!/usr/bin/env python3

import sys
import torch
import numpy as np
import ganutils

from architecture import load_generator, load_critic
from torch.optim import RMSprop, Adam
from tqdm import tqdm
from loader import cifar

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

    # MLP hyperparams
    'layer_size': 512,
}

# Informatation for argparse
INFO = {
    'critic_arch': [str, 'Architecture to use in Critic.'],
    'generator_arch': [str, 'Architecture to use in Generator.'],
    'zdim': [int, 'The dimension of latent vectors.'],
    'image_size': [int, 'Image size for dataset (must be power of 2).'],
    'nchannels': [int, 'Number of channels in image.'],
    'nfeatures': [int, 'Number of features in first layer of DCGAN.'],
    'clipping_constant': [float, 'Clipping constant for Wasserstein distance.'],
    'ncritic': [int, 'Number of times to train critic before generator.'],
    'gp_enabled': [bool, 'Whether or not to use gradient penalty.'],
    'gradient_penalty': [float, 'Gradient penalty constant for WGAN-GP.'],
    'layer_size': [int, 'Layer size for mlp.']
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

        # Start the GAN on cpu
        self.dev = torch.device('cpu')

        # Build critic and generator
        self.D = load_critic(self.S).to(self.dev)
        self.G = load_generator(self.S).to(self.dev)

    def __repr__(self):
        return f'GAN(\n{repr(self.D)}\n{repr(self.G)}\n)'

    def train(self, dl, iterations=1000, lr=0.0002, si=20, bs=128):
        ''' Trains the GAN on the given data '''
        if self.S['gp_enabled']:
            for metrics in self.train_gp(dl, iterations, lr, si, bs):
                yield metrics
        else:
            for metrics in self.train_no_gp(dl, iterations, lr, si, bs):
                yield metrics

    def train_gp(self, dl, iterations=1000, lr=0.0002, si=20, bs=128):
        ''' Trains the GAN on the given data using wasserstein with gp'''
        pass

    def train_no_gp(self, dl, iterations=1000, ci=0, lr=0.0002, si=20, bs=128):
        ''' Trains the GAN on the given data using vanilla wasserstein'''

        d_optim = RMSprop(self.D.parameters(), lr=lr)
        g_optim = RMSprop(self.G.parameters(), lr=lr)

        # Iterations of batches
        for iteration in tqdm(range(ci, iterations), ascii=True, initial=ci,
                              total=iterations):
            # Train the critic the specified number of times before generator
            d_losses = []
            for _ in range(self.S['ncritic']):
                self.D.zero_grad()
                x_batch = next(iter(dl)).to(self.dev)
                z_batch = self.get_latent_vec(bs).to(self.dev)
                g_out = self.G(z_batch).detach()
                d_real = self.D(x_batch)
                d_fake = self.D(g_out)
                d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
                d_loss.backward()
                d_optim.step()
                d_losses.append(d_loss)
                # Replaced in gp
                for p in self.D.parameters():
                    p.data = torch.clamp(p.data, -self.S['clipping_constant'],
                                         self.S['clipping_constant'])

            # Train the generator
            self.G.zero_grad()
            z_batch = self.get_latent_vec(bs).to(self.dev)
            g_out = self.G(z_batch)
            d_fake = self.D(g_out)
            g_loss = -torch.mean(d_fake)
            g_loss.backward()
            g_optim.step()

            # Yield results each sample inteval
            if iteration % si == 0:
                d_avg_loss = np.mean([l.cpu().item() for l in d_losses])
                g_loss_cpu = g_loss.cpu().item()
                yield {'d_loss': d_avg_loss, 'g_loss': g_loss_cpu}

    def get_latent_vec(self, n):
        ''' Returns random n x zdim x 1 x 1 latent vector '''
        return torch.randn(n, self.S['zdim'], 1, 1)

    def generate_image(self, z):
        ''' Generate an image from the generator with given latent vec '''
        return self.G(z).to(self.dev)

    def cpu(self):
        ''' Move the GAN to the cpu '''
        self.dev = torch.device('cpu')
        self.D.to(self.dev)
        self.G.to(self.dev)

    def cuda(self):
        ''' Move the GAN to the gpu '''
        if not torch.cuda.is_available():
            sys.stderr.write("Error: Cuda unavailable!\n")
            return
        self.dev = torch.device('cuda')
        self.D.to(self.dev)
        self.G.to(self.dev)


if __name__ == "__main__":

    name = 'wgan-test'
    dl = cifar.load(batch_size=64)

    if ganutils.trainer.is_training_started(name):
        ts, cp, res = ganutils.trainer.recover_training_state('wgan-test')
        gan = GAN(cp['settings'])
        gan.D.load_state_dict(cp['d_state_dict'])
        gan.G.load_state_dict(cp['g_state_dict'])
        gan.cuda()
        res = ganutils.trainer.train(gan, 'wgan-test', dl, ts,
                                     ci=cp['iteration'], cr=res)
    else:
        gan = GAN({
            'image_size': 32,
            'nchannels': 3,
            'nfeatures': 64
        })

        gan.cuda()

        res = ganutils.trainer.train(gan, 'wgan-test', dl, {
            'iterations': 300000,
            'sample_interval': 600,
            'learning_rate': 0.00005,
            'batch_size': 64
        })
