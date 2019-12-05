#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys
import imageio
import torch
import os
import random
from tqdm import tqdm

if __name__ == 'ganutils.visualize':
    from .utils import clean_images
    from . import trainer
else:
    from utils import clean_images
    import trainer


def plot_losses(d_losses: list, g_losses: list, name='GAN Losses'):
    ''' Plot GAN critic and generator losses '''

    plt.plot(d_losses, label='Critic')
    plt.plot(g_losses, label='Generator')
    plt.legend()
    plt.title(name)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()


def images_as_grid(images, rows, cols, name='grid', save=False):
    ''' Show a list of images as a rows x cols grid '''

    # Ensure we have the proper number of images
    n, imsize, _, channels = images.shape
    if rows * cols != n:
        sys.stderr.write("Error: Invalid dimensions for grid.\n")
        return

    # Reshape the list into a grid
    a = images.reshape(rows, cols, imsize, imsize, channels)
    a = a.swapaxes(1, 2)
    a = a.reshape(rows * imsize, cols * imsize, channels)
    if channels == 1:
        a = np.mean(a)

    # Show or save the grid of images
    if save:
        imageio.imwrite(f'{name}.png', a)
    else:
        plt.imshow(a)
        plt.axis('off')
        plt.title(name)
        plt.show()

       
def get_direction():
    sample_size = random.randint(50,100)
    return random.sample(range(100), sample_size)


def explore_dimensions(gan, rows=4, cols=4):
    """ Exploring the latent space """
    if rows * cols > 100:
        sys.stderr('Error trying to visualize too many dimensions.')
        exit()
    im_size = gan.S['image_size']
    z = torch.zeros(rows*cols,100, 1, 1).to(gan.dev) + 0.1
    # Save our frames in a list
    images = list()
    directions = [get_direction() for x in range(rows*cols)]
    mult = 2

    for i in tqdm(range(100)):
        with torch.no_grad():
            frame = list()
            frame = gan.G(z).cpu().numpy()
            frame = clean_images(frame)
            # Each frame of the gif will be a grid of multiple images
            frame = frame[:cols*rows].reshape(rows, cols, im_size, im_size, 3)
            frame = frame.swapaxes(1, 2)
            frame = frame.reshape(im_size * rows, im_size * cols, 3)
            images.append(frame)
            if i == 0:
                imageio.imsave('0.png', frame)
            if i == 1:
                imageio.imsave('1.png', frame)
        for i in range(rows*cols):
            for d in directions[i]:
                z[i,d] += 0.01
    return images

if __name__ == '__main__':

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import artgan

    results = trainer.load_results('ross-wgan-1')

    gan = artgan.GAN(results['settings'])
    gan.D.load_state_dict(results['d_state_dict'])
    gan.G.load_state_dict(results['g_state_dict'])
    gan.cuda()
    imageio.mimsave('test.gif', explore_dimensions(gan, rows=10, cols=10), duration=0.1)
