#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys
import imageio
import torch
import os
import random
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import artgan

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
    sample_size = random.randint(1,100)
    return random.sample(range(100), sample_size)


def explore_dimensions(gan, path, rand_start=True, rows=4, cols=4, mult=0.002):
    """ Exploring the latent space """
    print(f'Generating image: {path}')
    if rows * cols > 100:
        sys.stderr('Error trying to visualize too many dimensions.')
        exit()
    im_size = gan.S['image_size']
    if rand_start:
        z = torch.ones(rows*cols,100, 1, 1).to(gan.dev) * torch.randn(100,1,1).to(gan.dev)
    else:
        z = torch.zeros(rows*cols,100, 1, 1).to(gan.dev) + 0.01

    rz = torch.randn(rows*cols,100, 1, 1).to(gan.dev) * mult
    # Save our frames in a list
    images = list()
    directions = [get_direction() for x in range(rows*cols)]

    for i in range(50):
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
            z += rz * mult
    imageio.mimsave(path, images, duration=0.1)

def explore(name: str, rand_start=False, rows=5, cols=5, num=16):
    import threading
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results',
        'exploration'
        )
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print(out_dir)
    gan = load_gan()
    threads = list()
    for i in range(num):
        path = os.path.join(out_dir, f'{i:02d}.gif')
        thread = threading.Thread(target=explore_dimensions, args=(gan, path, rand_start, rows, cols))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

def generate_image(name: str, z):
    with torch.no_grad():
        settings, gan = load_gan(name)
        z = z.to(gan.dev)
        ts = datetime.now()
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'images',
            f'{ts}.png'
            )
        image = gan.G(z).cpu().numpy()
        image = clean_images(image)
        # imageio.imsave(path, image)
        return image




def load_gan(name: str):
    """ I'll probably move this somewhere else in the future """
    results = trainer.load_results(name)
    gan = artgan.GAN(results['settings'])
    gan.D.load_state_dict(results['d_state_dict'])
    gan.G.load_state_dict(results['g_state_dict'])
    gan.cuda()
    return results['settings'], gan


if __name__ == '__main__':
    # explore('ross-wgan-1', rand_start=False)
    # imageio.mimsave('test.gif', explore_dimensions(gan, rows=10, cols=10), duration=0.1)
    plt.imshow(generate_image('ross-wgan-1', torch.randn(1, 100,1,1))[0])
    plt.show()
