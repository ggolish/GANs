
import maplotlib.pyplot as plt
import numpy as np
import sys
import imageio


def plot_losses(d_losses: list, g_losses: list, name='GAN Losses'):
    ''' Plot GAN critic and generator losses '''

    plt.plot(d_losses, 'Critic')
    plt.plot(g_losses, 'Generator')
    plt.legend()
    plt.title(name)
    plt.show()


def images_as_grid(images, rows, cols, name='grid', save=False):
    ''' Show a list of images as a rows x cols grid '''

    # Ensure we have the proper number of images
    n, imsize, _, channels = images.shape
    if rows * cols != n:
        sys.sterr.write("Error: Invalid dimensions for grid.\n")
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
